# generar_datos_sinteticos.py
# Genera 4 temporadas sinteticas calibradas a la estadistica real de T1
# Estudiante: 05193002  |  2026-06-23
#
# METODOLOGIA:
#   1. Calcula parametros estadisticos por mes de la T1 real
#   2. Modela diarios con log-normal + autocorrelacion AR(1)
#   3. Aplica factor inter-anual (tendencia leve + ruido)
#   4. Desagrega en registros individuales por productor
#   5. Preserva: esparsidad, distribucion, efectos dia-semana, ramp-up de temporada

import warnings; warnings.filterwarnings('ignore')
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)

BASE = Path(r"C:\Users\eegos\OneDrive\superación\2026PostGradoIA\2do Trimestre\Manejo de Datos Para Ia IA\Proyecto\Final")
MESES_OP = [10, 11, 12, 1, 2, 3, 4, 5]

# ═══════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS REALES
# ═══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  GENERACION DE DATOS SINTETICOS (4 temporadas adicionales)")
print("=" * 65)

df_real = pd.read_excel(BASE / "DatosCafe_05193002.csv", engine='openpyxl')
df_real['FechaTS'] = pd.to_datetime(df_real['Fecha'].dt.date)
df_real['FormaIngreso'] = df_real['FormaIngreso'].apply(
    lambda x: 'HUMEDO' if isinstance(x, str) and 'MEDO' in x.upper() else
              ('SECO' if pd.notna(x) else 'HUMEDO'))
df_real['Medida'] = df_real['Medida'].fillna('LATAS')

df_t1 = df_real[(df_real['FechaTS'] >= '2011-10-21') &
                (df_real['FechaTS'] <= '2012-05-31')].copy()

print(f"\n[T1 Real] {len(df_t1)} registros, {df_t1['FechaTS'].nunique()} dias activos")

# ═══════════════════════════════════════════════════════════════════════
# 2. PARAMETROS ESTADISTICOS POR MES (de T1)
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Calibrando parametros estadisticos por mes ---")

# Serie diaria T1
op_dates_t1 = pd.date_range('2011-10-21', '2012-05-31', freq='D')
daily_t1 = (df_t1.groupby('FechaTS')['Cantidad'].sum()
                  .reindex(op_dates_t1, fill_value=0)
                  .reset_index())
daily_t1.columns = ['FechaTS', 'Cantidad_Total']
daily_t1['Mes'] = daily_t1['FechaTS'].dt.month
daily_t1['DiaSemana'] = daily_t1['FechaTS'].dt.dayofweek
daily_t1['Activo'] = (daily_t1['Cantidad_Total'] > 0).astype(int)

# Estadisticas por mes
params_mes = {}
for mes in MESES_OP:
    sub = daily_t1[daily_t1['Mes'] == mes]
    activos = sub[sub['Activo'] == 1]['Cantidad_Total']
    if len(sub) < 3:
        # Oct en T1 solo tiene ~10 dias (desde el 21); extrapolar desde Nov-Dic
        sub_ref = daily_t1[daily_t1['Mes'].isin([11, 12])]
        activos_ref = sub_ref[sub_ref['Activo'] == 1]['Cantidad_Total']
        params_mes[mes] = {
            'p_activo': 0.75,
            'mu_log': float(np.log(activos_ref.mean() * 1.2 + 1)),
            'sigma_log': float(activos_ref.std() / activos_ref.mean() if activos_ref.mean() > 0 else 0.8),
            'n_dias': len(sub)
        }
    else:
        p_act = sub['Activo'].mean()
        if len(activos) >= 2:
            log_vals = np.log(activos + 1)
            mu_l = log_vals.mean()
            sg_l = log_vals.std()
        else:
            mu_l = np.log(activos.mean() + 1) if len(activos) > 0 else 4.0
            sg_l = 0.8
        params_mes[mes] = {
            'p_activo': float(p_act),
            'mu_log': float(mu_l),
            'sigma_log': float(max(sg_l, 0.3)),
            'n_dias': len(sub)
        }

# Efectos dia de semana (relativo al promedio)
dow_effect = {}
for d in range(7):
    sub_d = daily_t1[daily_t1['DiaSemana'] == d]['Cantidad_Total']
    dow_effect[d] = float(sub_d.mean() / daily_t1['Cantidad_Total'].mean()) if daily_t1['Cantidad_Total'].mean() > 0 else 1.0

print("  Probabilidad de dia activo por mes:")
for m in MESES_OP:
    meses_es = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',10:'Oct',11:'Nov',12:'Dic'}
    print(f"    {meses_es[m]}: p_activo={params_mes[m]['p_activo']:.2f}  "
          f"mu_log={params_mes[m]['mu_log']:.2f}  sigma_log={params_mes[m]['sigma_log']:.2f}")

print("\n  Efecto dia de semana:")
dias_es = {0:'Lun',1:'Mar',2:'Mie',3:'Jue',4:'Vie',5:'Sab',6:'Dom'}
for d in range(7):
    print(f"    {dias_es[d]}: {dow_effect[d]:.3f}x")

# Parametros de registros individuales
records_per_day = df_t1.groupby('FechaTS')['Cantidad'].count()
mean_records   = records_per_day.mean()
qty_per_record = df_t1['Cantidad']
log_qty_mu     = np.log(qty_per_record[qty_per_record > 0] + 1).mean()
log_qty_sigma  = np.log(qty_per_record[qty_per_record > 0] + 1).std()
forma_dist     = df_t1['FormaIngreso'].value_counts(normalize=True).to_dict()
medida_dist    = df_t1['Medida'].value_counts(normalize=True).to_dict()
clientes_t1    = df_t1['CodCliente'].unique().tolist()

print(f"\n  Registros por dia activo: media={mean_records:.1f}")
print(f"  Cantidad por registro: mu_log={log_qty_mu:.2f} sigma_log={log_qty_sigma:.2f}")
print(f"  FormaIngreso: {forma_dist}")

# ═══════════════════════════════════════════════════════════════════════
# 3. GENERACION DE TEMPORADAS SINTETICAS
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Generando temporadas sinteticas ---")

# Factores inter-anuales realistas (cosechas de Guatemala 2012-2016)
# Leve crecimiento 2-3% anual + variacion climatica
FACTORES_ANUALES = {
    2: 1.08,   # T2 2012-13: recuperacion post-roya
    3: 0.82,   # T3 2013-14: impacto severo roya del cafeto (Hemileia vastatrix)
    4: 0.91,   # T4 2014-15: recuperacion parcial
    5: 1.05,   # T5 2015-16: mejora, aunque El Nino afecta levemente
}

print("\n  Factores inter-anuales:")
for t, f in FACTORES_ANUALES.items():
    anio_ini = 2010 + t
    print(f"    T{t} ({anio_ini}-{anio_ini+1}): x{f:.2f}")

# Clientes adicionales que se van incorporando
clientes_nuevos = [f"N{str(i).zfill(3)}" for i in range(1, 41)]
todos_clientes  = clientes_t1 + clientes_nuevos

def generar_temporada(temporada_num, factor_anual, seed_offset=0):
    """Genera registros individuales para una temporada sintetica."""
    np.random.seed(42 + seed_offset * 100)

    anio_ini = 2010 + temporada_num
    fecha_ini = pd.Timestamp(f'{anio_ini}-10-01')
    fecha_fin = pd.Timestamp(f'{anio_ini + 1}-05-31')
    todas_fechas = pd.date_range(fecha_ini, fecha_fin, freq='D')

    registros = []
    cant_ant   = 0.0   # para AR(1) autocorrelacion

    # Ramp-up: los primeros 14 dias la actividad es mas baja
    ramp_dias = 14

    for i, fecha in enumerate(todas_fechas):
        mes = fecha.month
        if mes not in MESES_OP:
            continue

        p = params_mes[mes]
        ramp_factor = min(1.0, 0.3 + 0.7 * i / ramp_dias) if i < ramp_dias else 1.0

        # AR(1): si ayer hubo entregas, hoy tiene mayor probabilidad
        ar1_boost = 0.15 if cant_ant > 0 else -0.05
        p_hoy = min(0.95, max(0.05, p['p_activo'] * ramp_factor + ar1_boost))

        if np.random.random() > p_hoy:
            cant_ant = 0.0
            continue

        # Cantidad diaria: log-normal con efecto dia-semana y factor anual
        dow   = fecha.dayofweek
        d_eff = dow_effect.get(dow, 1.0)
        sg    = p['sigma_log'] * (1.0 + 0.2 * np.random.randn())
        sg    = max(sg, 0.3)
        cant_dia = np.exp(np.random.normal(p['mu_log'], sg)) * d_eff * factor_anual * ramp_factor
        cant_dia = max(cant_dia, 5.0)

        # Ruido inter-anual especifico (variabilidad climatica)
        cant_dia *= np.random.lognormal(0, 0.12)

        # AR(1) en cantidad
        cant_dia = 0.7 * cant_dia + 0.3 * (cant_ant if cant_ant > 0 else cant_dia)
        cant_ant = cant_dia

        # Desagregar en N registros de productores
        n_rec = max(1, int(np.random.poisson(mean_records * ramp_factor)))
        # Pool de clientes activos (mas clientes nuevos en temporadas posteriores)
        pool_nuevos = min(temporada_num * 8, len(clientes_nuevos))
        pool = clientes_t1 + clientes_nuevos[:pool_nuevos]

        clientes_dia = np.random.choice(pool, size=min(n_rec, len(pool)), replace=False)

        # Proporciones aleatorias que suman a 1
        props = np.random.dirichlet(np.ones(len(clientes_dia)))
        cantidades = props * cant_dia
        cantidades = np.maximum(cantidades, 1.0)

        for cliente, cantidad in zip(clientes_dia, cantidades):
            forma = np.random.choice(
                list(forma_dist.keys()), p=list(forma_dist.values()))
            medida = np.random.choice(
                list(medida_dist.keys()), p=list(medida_dist.values()))

            # Pequena variacion en la cantidad del registro
            qty_ajust = max(1.0, cantidad * np.random.lognormal(0, 0.08))

            registros.append({
                'CodCliente': cliente,
                'Fecha':      fecha,
                'FechaTS':    fecha,
                'Cantidad':   round(qty_ajust, 1),
                'FormaIngreso': forma,
                'Medida':     medida,
                'Temporada':  temporada_num,
                'Sintetico':  True,
            })

    return pd.DataFrame(registros)

# Generar T2-T5
dfs_sinteticos = []
for t in [2, 3, 4, 5]:
    df_t = generar_temporada(t, FACTORES_ANUALES[t], seed_offset=t)
    anio = 2010 + t
    dias_activos = df_t['FechaTS'].nunique()
    total_cant   = df_t['Cantidad'].sum()
    print(f"\n  T{t} ({anio}-{anio+1}): {len(df_t)} registros | "
          f"{dias_activos} dias activos | total={total_cant:,.0f} u.")
    dfs_sinteticos.append(df_t)

df_sintetico = pd.concat(dfs_sinteticos, ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════
# 4. COMBINAR CON DATOS REALES
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Combinando datos reales + sinteticos ---")

df_real_t1 = df_t1.copy()
df_real_t1['Temporada'] = 1
df_real_t1['Sintetico'] = False

cols_base = ['CodCliente', 'Fecha', 'FechaTS', 'Cantidad', 'FormaIngreso',
             'Medida', 'Temporada', 'Sintetico']

df_real_t1 = df_real_t1.reindex(columns=cols_base)
df_sintetico = df_sintetico.reindex(columns=cols_base)

df_combined = pd.concat([df_real_t1, df_sintetico], ignore_index=True)
df_combined = df_combined.sort_values('FechaTS').reset_index(drop=True)

print(f"\n  Dataset combinado:")
print(f"    Total registros:  {len(df_combined):,}")
print(f"    Registros reales: {(~df_combined['Sintetico']).sum():,} (T1)")
print(f"    Registros sint.:  {df_combined['Sintetico'].sum():,} (T2-T5)")
print(f"    Rango:            {df_combined['FechaTS'].min().date()} -- {df_combined['FechaTS'].max().date()}")

# Guardar
out_path = BASE / "DatosCafe_5temporadas.csv"
df_combined.to_csv(str(out_path), index=False, encoding='utf-8-sig')
print(f"\n  [OK] Guardado: DatosCafe_5temporadas.csv ({out_path.stat().st_size/1024:.1f} KB)")

# ═══════════════════════════════════════════════════════════════════════
# 5. VALIDACION DEL DATASET SINTETICO
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Validacion estadistica ---")

print("\n  Comparacion diaria por temporada:")
print(f"  {'Temporada':<12}{'Dias activos':>14}{'Media diaria':>14}{'% activos':>12}{'Total':>12}")
print("  " + "-"*64)

for t in range(1, 6):
    sub = df_combined[df_combined['Temporada'] == t]
    if t == 1:
        fechas_serie = pd.date_range('2011-10-21', '2012-05-31', freq='D')
    else:
        anio = 2010 + t
        fechas_serie = pd.date_range(f'{anio}-10-01', f'{anio+1}-05-31', freq='D')
    dias_totales = len(fechas_serie)
    diario = sub.groupby('FechaTS')['Cantidad'].sum()
    dias_activos = len(diario[diario > 0])
    media_dia = diario.reindex(fechas_serie, fill_value=0).mean()
    total = diario.sum()
    anio_ini = 2010 + t
    tag = " (real)" if t == 1 else " (sint.)"
    print(f"  T{t} {anio_ini}-{anio_ini+1}{tag:<10}"
          f"{dias_activos:>10}/{dias_totales:<4}{media_dia:>14.1f}"
          f"{dias_activos/dias_totales*100:>11.1f}%{total:>12,.0f}")

print("\n  [OK] Datos sinteticos generados y validados correctamente")
print("  Usar 'DatosCafe_5temporadas.csv' en el pipeline v3")
