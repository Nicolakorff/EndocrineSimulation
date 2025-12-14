# Modelo y tokenizer
MODEL_NAME = "gpt2-medium"  # Alternativas: "gpt2", "gpt2-medium"

print(f"🔄 Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Configuración del tokenizer
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"✅ Model loaded successfully")
print(f"   Parameters: {model.num_parameters() / 1e6:.1f}M")
print(f"   Vocabulary size: {len(tokenizer)}")

# Lista inicial de palabras empáticas (expandir según necesidad)
EMPATHETIC_WORDS = [
    "understand", "care", "help", "support", "feel", "sorry", "comfort",
    "listen", "empathy", "compassion", "kindness", "gentle", "warm",
    "hope", "better", "together", "here", "okay", "safe"
]

# Convertir a token IDs (incluyendo variantes con y sin espacio inicial)
empathetic_token_ids = []
for word in EMPATHETIC_WORDS:
    # Variante con espacio previo (común en GPT)
    tokens_space = tokenizer.encode(" " + word, add_special_tokens=False)
    empathetic_token_ids.extend(tokens_space)
    # Variante sin espacio (inicio de frase o composición con otros sufijos)
    tokens_no_space = tokenizer.encode(word, add_special_tokens=False)
    empathetic_token_ids.extend(tokens_no_space)

# Remover duplicados
empathetic_token_ids = list(set(empathetic_token_ids))

print(f"📝 Lexicón empático: {len(empathetic_token_ids)} tokens únicos")
print(f"   Ejemplos: {[tokenizer.decode([tid]) for tid in empathetic_token_ids[:10]]}")

def generate_with_hormones(
    prompt: str,
    hormonal_vector: HormonalVector,
    max_new_tokens: int = 50,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
) -> List[str]:
    """
    Genera texto con modulación hormonal.

    Args:
        prompt: texto de entrada
        hormonal_vector: configuración hormonal
        max_new_tokens: longitud máxima de generación
        num_return_sequences: número de muestras independientes
        do_sample: si se usa muestreo (True) o greedy (False)
        top_k: recorte clásico de vocabulario (se combina con la adrenalina)
        top_p: nucleus sampling

    Returns:
        Lista de textos generados.
    """
    # Construir logits processors
    hormonal_processor = HormonalLogitsProcessor(
        hormonal_vector=hormonal_vector,
        empathetic_tokens=empathetic_token_ids,
    )

    processors = [hormonal_processor]

    # Si el cortisol es alto, añadimos explícitamente un NoRepeatNGramLogitsProcessor
    # para evitar bucles repetitivos cuando la temperatura baja demasiado.
    if hormonal_vector.cortisol >= 0.6:
        processors.append(NoRepeatNGramLogitsProcessor(3))

    logits_processor = LogitsProcessorList(processors)

    # Preparar el prompt
    # he modificado aquí el tema del padding
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}


    # Generación
    # he modificado aqui para añadir la attention_mask
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ) 


    # Decodificar
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts

print("✅ Función de generación lista")
