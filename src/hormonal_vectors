@dataclass
class HormonalVector:
    """Vector hormonal H = [d, c, o, a, s]"""
    dopamine: float = 0.5      # d: exploración
    cortisol: float = 0.5      # c: cautela
    oxytocin: float = 0.5      # o: prosocialidad
    adrenaline: float = 0.5    # a: activación
    serotonin: float = 0.5     # s: estabilidad

    def __repr__(self):
        return f"H[d={self.dopamine:.2f}, c={self.cortisol:.2f}, o={self.oxytocin:.2f}, a={self.adrenaline:.2f}, s={self.serotonin:.2f}]"

    def to_dict(self):
        return {
            'dopamine': self.dopamine,
            'cortisol': self.cortisol,
            'oxytocin': self.oxytocin,
            'adrenaline': self.adrenaline,
            'serotonin': self.serotonin
        }

# Perfiles hormonales predefinidos
PROFILES = {
    "baseline": HormonalVector(0.5, 0.5, 0.5, 0.5, 0.5),
    "euphoric": HormonalVector(dopamine=0.9, serotonin=0.8, cortisol=0.2, adrenaline=0.5, oxytocin=0.5),
    "stressed": HormonalVector(cortisol=0.9, adrenaline=0.7, serotonin=0.3, dopamine=0.5, oxytocin=0.5),
    "empathetic": HormonalVector(oxytocin=0.9, serotonin=0.7, dopamine=0.6, cortisol=0.5, adrenaline=0.5),
    "cautious": HormonalVector(cortisol=0.7, dopamine=0.3, serotonin=0.6, adrenaline=0.5, oxytocin=0.5),
    "high_dopamine": HormonalVector(dopamine=0.9, cortisol=0.5, oxytocin=0.5, adrenaline=0.5, serotonin=0.5),
    "high_serotonin": HormonalVector(serotonin=0.9, dopamine=0.5, cortisol=0.5, adrenaline=0.5, oxytocin=0.5),
}

print("📊 Perfiles hormonales disponibles:")
for name, profile in PROFILES.items():
    print(f"  - {name}: {profile}")
