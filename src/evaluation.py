# Ejecutar experimento comparativo
def run_comparative_experiment(
    prompts: List[str],
    profiles: Dict[str, HormonalVector],
    max_new_tokens: int = 40,
) -> pd.DataFrame:
    """
    Ejecuta generación con múltiples perfiles y prompts.

    Returns:
        DataFrame con resultados
    """
    results = []

    for prompt in tqdm(prompts, desc="Prompts"):
        for profile_name, hormonal_vec in profiles.items():
            outputs = generate_with_hormones(
                prompt=prompt,
                hormonal_vector=hormonal_vec,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1
            )

            results.append({
                'prompt': prompt,
                'profile': profile_name,
                'hormonal_vector': str(hormonal_vec),
                'generated_text': outputs[0],
                'length': len(tokenizer.encode(outputs[0]))
            })

    return pd.DataFrame(results)

# Ejecutar experimento
print("🚀 Ejecutando experimento comparativo...")
experiment_df = run_comparative_experiment(
    prompts=TEST_PROMPTS[:6],  # Usar subset para prueba rápida
    profiles={k: PROFILES[k] for k in ["baseline", "empathetic", "cautious"]},
    max_new_tokens=30
)

print(f"\n✅ Experimento completado: {len(experiment_df)} generaciones")
experiment_df.head(10)

from collections import Counter

# Modelo ligero de análisis de sentimiento (Transformer)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

def compute_basic_metrics(text: str) -> Dict[str, float]:
    """
    Calcula métricas básicas del texto generado.

    Métricas:
    - distinct_1 / distinct_2: diversidad léxica (unigramas / bigramas)
    - sentiment_polarity: sentimiento global en [-1, 1]
    - sentiment_subjectivity: confianza del clasificador en [0, 1]
    - repetition_rate: tasa de repetición aproximada
    """
    tokens = tokenizer.encode(text)

    # Distinct-1 y Distinct-2 (diversidad léxica)
    words = text.lower().split()
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]

    distinct_1 = len(set(words)) / max(len(words), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)

    # Longitud media de frase (aproximada)
    sentences = [s for s in text.split(".") if s.strip()]
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0.0

    # Sentimiento con Transformer
    try:
        # Limitar longitud para evitar problemas de memoria
        result = sentiment_analyzer(text[:512])[0]
        # label: POSITIVE / NEGATIVE, score: [0, 1]
        polarity = result["score"] if result["label"].upper().startswith("POS") else -result["score"]
        subjectivity = result["score"]  # interpretamos la confianza como "intensidad subjetiva"
    except Exception:
        polarity = 0.0
        subjectivity = 0.0

    # Repetición aproximada: porcentaje de palabras que aparecen más de una vez
    word_counts = Counter(words)
    repeated_tokens = sum(count for count in word_counts.values() if count > 1)
    repetition_rate = repeated_tokens / max(len(words), 1)

    return {
        'distinct_1': float(distinct_1),
        'distinct_2': float(distinct_2),
        'avg_sentence_length': float(avg_sentence_length),
        'sentiment_polarity': float(polarity),
        'sentiment_subjectivity': float(subjectivity),
        'repetition_rate': float(repetition_rate),
    }

# Aplicar métricas al DataFrame
print("📊 Calculando métricas...")
metrics_list = []
for _, row in tqdm(experiment_df.iterrows(), total=len(experiment_df)):
    metrics = compute_basic_metrics(row['generated_text'])
    metrics['prompt'] = row['prompt']
    metrics['profile'] = row['profile']
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
print("\n✅ Métricas calculadas")
metrics_df.head()

# Agrupar por perfil y calcular estadísticas
profile_stats = metrics_df.groupby('profile').agg({
    'distinct_1': ['mean', 'std'],
    'distinct_2': ['mean', 'std'],
    'sentiment_polarity': ['mean', 'std'],
    'sentiment_subjectivity': ['mean', 'std'],
    'repetition_rate': ['mean', 'std'],
}).round(3)

print("📈 Estadísticas por perfil hormonal:\n")
print(profile_stats)

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🧠 Análisis Comparativo: Perfiles Hormonales', fontsize=16, fontweight='bold')

profiles = sorted(metrics_df['profile'].unique())

def add_strip(ax, metric_name):
    """Superpone puntos individuales sobre el boxplot para ver cuántas muestras hay."""
    # Posición base 1..N, asumimos orden alfabético
    x_pos = {profile: i + 1 for i, profile in enumerate(profiles)}
    for profile, group in metrics_df.groupby('profile'):
        x = np.random.normal(loc=x_pos[profile], scale=0.05, size=len(group))
        y = group[metric_name].values
        ax.scatter(x, y, alpha=0.6, s=10)

# 1. Diversidad léxica (Distinct-1)
metrics_df.boxplot(column='distinct_1', by='profile', ax=axes[0, 0])
add_strip(axes[0, 0], 'distinct_1')
axes[0, 0].set_title('Diversidad Léxica (Distinct-1)')
axes[0, 0].set_xlabel('Perfil Hormonal')
axes[0, 0].set_ylabel('Distinct-1 Score')

# 2. Sentimiento (polaridad)
metrics_df.boxplot(column='sentiment_polarity', by='profile', ax=axes[0, 1])
add_strip(axes[0, 1], 'sentiment_polarity')
axes[0, 1].set_title('Sentimiento (Polaridad)')
axes[0, 1].set_xlabel('Perfil Hormonal')
axes[0, 1].set_ylabel('Polarity (-1 a 1)')

# 3. Subjetividad
metrics_df.boxplot(column='sentiment_subjectivity', by='profile', ax=axes[1, 0])
add_strip(axes[1, 0], 'sentiment_subjectivity')
axes[1, 0].set_title('Subjetividad / Intensidad emocional')
axes[1, 0].set_xlabel('Perfil Hormonal')
axes[1, 0].set_ylabel('Score (0 a 1)')

# 4. Tasa de repetición
metrics_df.boxplot(column='repetition_rate', by='profile', ax=axes[1, 1])
add_strip(axes[1, 1], 'repetition_rate')
axes[1, 1].set_title('Tasa de Repetición')
axes[1, 1].set_xlabel('Perfil Hormonal')
axes[1, 1].set_ylabel('Repetition Rate')

plt.tight_layout()
plt.show()
