class HormonalLogitsProcessor(LogitsProcessor):
    """
    Procesador de logits que implementa modulación endocrina.

    Aplica tres transformaciones principales:

    1. **Temperatura adaptativa**:
       T = T_base + α·dopamina − β·cortisol

       - Dopamina ↑  ⇒ T sube ⇒ más exploración.
       - Cortisol ↑  ⇒ T baja ⇒ más foco en tokens muy probables.

    2. **Moldeado de la distribución**:
       - Serotonina: suaviza extremos (reduce la varianza de los logits).
       - Adrenalina: si es alta, aplica un **Top-K dinámico** que reduce el
         vocabulario disponible (visión de túnel).

    3. **Sesgo prosocial (oxitocina)**:
       - Potencia suavemente la probabilidad de tokens empáticos predefinidos.

    La idea es que cada hormona tenga un efecto cualitativamente distinto y
    no sea simplemente “otra temperatura disfrazada”.
    """

    def __init__(
        self,
        hormonal_vector: HormonalVector,
        base_temperature: float = 1.0,
        alpha: float = 0.3,   # peso dopamina
        beta: float = 0.3,    # peso cortisol
        gamma: float = 0.2,   # fuerza del Top-K dinámico (adrenalina)
        delta: float = 0.2,   # fuerza de suavizado (serotonina)
        epsilon: float = 0.5, # sesgo oxitocina
        empathetic_tokens: List[int] = None,
        adrenaline_threshold: float = 0.6,
        oxytocin_threshold: float = 0.6,
    ):
        self.H = hormonal_vector
        self.T_base = base_temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.empathetic_tokens = empathetic_tokens or []
        self.adrenaline_threshold = adrenaline_threshold
        self.oxytocin_threshold = oxytocin_threshold

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Aplica modulación hormonal a los logits.

        Args:
            input_ids: IDs de tokens generados hasta ahora
            scores: Logits del modelo (batch_size, vocab_size)

        Returns:
            scores modificados
        """
        # 1. Temperatura adaptativa
        T_adaptive = self.T_base + self.alpha * self.H.dopamine - self.beta * self.H.cortisol
        # Evitar temperaturas numéricamente peligrosas
        T_adaptive = max(0.1, float(T_adaptive))
        scores = scores / T_adaptive

        # 2a. Serotonina: suavizado de extremos (reduce la varianza de los logits)
        if self.H.serotonin > 0:
            mean_scores = scores.mean(dim=-1, keepdim=True)
            # Mezcla entre logits originales y su media: cuanto más serotonina,
            # más se acercan los logits a la media (menos extremos).
            smoothing_factor = max(0.0, min(1.0, self.delta * self.H.serotonin))
            scores = mean_scores + (scores - mean_scores) * (1.0 - smoothing_factor)

        # 2b. Adrenalina: Top-K dinámico (visión de túnel)
        if self.H.adrenaline > self.adrenaline_threshold:
            batch_size, vocab_size = scores.shape
            max_k = min(50, vocab_size)
            min_k = min(5, max_k)

            # Normalizar adrenalina a [0,1] por encima del umbral
            a_norm = (self.H.adrenaline - self.adrenaline_threshold) / (1.0 - self.adrenaline_threshold + 1e-8)
            a_norm = max(0.0, min(1.0, a_norm))

            # Cuanta más adrenalina, más pequeño es K
            k = int(max_k - a_norm * (max_k - min_k))
            k = max(min_k, min(max_k, k))

            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
            # Construimos una máscara con -inf fuera del Top-K
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_indices, topk_scores)
            scores = mask

        # 3. Sesgo prosocial (oxitocina)
        if self.H.oxytocin > self.oxytocin_threshold and len(self.empathetic_tokens) > 0:
            for token_id in self.empathetic_tokens:
                if token_id < scores.shape[-1]:  # Verificar que esté en vocabulario
                    scores[:, token_id] += self.epsilon * self.H.oxytocin

        return scores

print("✅ HormonalLogitsProcessor definido")

