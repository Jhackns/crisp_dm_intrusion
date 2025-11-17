#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploración Inicial del Dataset NSL-KDD
Proyecto: Detección de Intrusiones en Redes
Metodología: CRISP-DM
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configurar pandas para mostrar todos los datos
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("EXPLORACIÓN INICIAL DEL DATASET NSL-KDD")
print("="*80)

# Nombres de las columnas según la documentación del NSL-KDD
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

print("\n1. CARGANDO DATASETS...")
print("-"*80)

# Cargar los datos
train_df = pd.read_csv('KDDTrain.txt', names=column_names, header=None)
test_df = pd.read_csv('KDDTest.txt', names=column_names, header=None)

print(f"✓ Dataset de entrenamiento cargado: {train_df.shape}")
print(f"✓ Dataset de prueba cargado: {test_df.shape}")

print("\n2. INFORMACIÓN GENERAL DEL DATASET DE ENTRENAMIENTO")
print("-"*80)
print(f"Número de registros: {len(train_df):,}")
print(f"Número de características: {len(train_df.columns)}")
print(f"\nPrimeras 10 filas del dataset:")
print(train_df.head(10))

print("\n3. TIPOS DE DATOS")
print("-"*80)
print(train_df.dtypes)

print("\n4. INFORMACIÓN BÁSICA")
print("-"*80)
train_df.info()

print("\n5. VALORES FALTANTES")
print("-"*80)
missing_values = train_df.isnull().sum()
missing_percentage = (missing_values / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Columna': missing_values.index,
    'Valores Faltantes': missing_values.values,
    'Porcentaje': missing_percentage.values
})
missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)

if len(missing_df) == 0:
    print("✓ No se encontraron valores faltantes en el dataset")
else:
    print(missing_df)

print("\n6. ESTADÍSTICAS DESCRIPTIVAS (Variables numéricas)")
print("-"*80)
print(train_df.describe())

print("\n7. ANÁLISIS DE LA VARIABLE OBJETIVO (LABEL)")
print("-"*80)
print("\nDistribución de etiquetas:")
label_counts = train_df['label'].value_counts()
print(label_counts)

print(f"\nNúmero total de etiquetas únicas: {train_df['label'].nunique()}")
print(f"\nEtiquetas únicas:")
print(sorted(train_df['label'].unique()))

# Categorizar ataques
print("\n8. CATEGORIZACIÓN DE ATAQUES")
print("-"*80)

# Crear categorías de ataques según la literatura
dos_attacks = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm']
probe_attacks = ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']
r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 
               'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop']
u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack', 'xterm', 'ps']

def categorize_attack(label):
    if label == 'normal':
        return 'Normal'
    elif label in dos_attacks:
        return 'DoS'
    elif label in probe_attacks:
        return 'Probe'
    elif label in r2l_attacks:
        return 'R2L'
    elif label in u2r_attacks:
        return 'U2R'
    else:
        return 'Unknown'

train_df['attack_category'] = train_df['label'].apply(categorize_attack)
test_df['attack_category'] = test_df['label'].apply(categorize_attack)

print("\nDistribución por categorías de ataque:")
category_counts = train_df['attack_category'].value_counts()
print(category_counts)
print(f"\nPorcentajes:")
print((category_counts / len(train_df) * 100).round(2))

print("\n9. VARIABLES CATEGÓRICAS")
print("-"*80)
categorical_cols = train_df.select_dtypes(include=['object']).columns
print(f"\nColumnas categóricas: {list(categorical_cols)}")

for col in categorical_cols:
    if col not in ['label', 'attack_category']:
        print(f"\n{col}: {train_df[col].nunique()} valores únicos")
        print(train_df[col].value_counts().head(10))

print("\n10. RESUMEN ESTADÍSTICO POR CATEGORÍA")
print("-"*80)

# Crear variable binaria
train_df['is_attack'] = (train_df['label'] != 'normal').astype(int)
test_df['is_attack'] = (test_df['label'] != 'normal').astype(int)

print("\nDistribución binaria (Normal vs Ataque):")
print(train_df['is_attack'].value_counts())
print(f"\nPorcentaje de ataques: {(train_df['is_attack'].sum() / len(train_df) * 100):.2f}%")

# Guardar los dataframes procesados
print("\n11. GUARDANDO DATASETS PROCESADOS...")
print("-"*80)
train_df.to_csv('train_processed.csv', index=False)
test_df.to_csv('test_processed.csv', index=False)
print("✓ Datasets guardados: train_processed.csv, test_processed.csv")

print("\n" + "="*80)
print("EXPLORACIÓN INICIAL COMPLETADA")
print("="*80)
