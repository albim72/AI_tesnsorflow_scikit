from __future__ import annotations
from itertools import product
from typing import Callable, Dict, List, Tuple

Number = float | int
Example = Tuple[Number, Number]


# --- Prymitywne „ruchy” Maga (bazowe zaklęcia) ---


def times2(x: Number) -> Number:
    return x * 2


def times3(x: Number) -> Number:
    return x * 3


def plus1(x: Number) -> Number:
    return x + 1


def plus2(x: Number) -> Number:
    return x + 2


PRIMITIVES: Dict[str, Callable[[Number], Number]] = {
    "times2": times2,
    "times3": times3,
    "plus1": plus1,
    "plus2": plus2,
}


# --- Silnik autoprogramowania Maga ---


def apply_sequence(seq: Tuple[str, ...], x: Number) -> Number:
    """Zastosuj sekwencję nazw prymitywów do wartości x."""
    v = x
    for name in seq:
        v = PRIMITIVES[name](v)
    return v


def fits_examples(seq: Tuple[str, ...], examples: List[Example]) -> bool:
    """Sprawdź, czy dana sekwencja realizuje wszystkie przykłady."""
    return all(apply_sequence(seq, x) == y for x, y in examples)


def synthesize_spell_from_examples(
    examples: List[Example],
    max_depth: int = 4,
    func_name: str = "emergent_spell",
) -> Callable[[Number], Number]:
    """
    Szuka sekwencji prymitywów, która spełnia wszystkie przykłady,
    a następnie GENERUJE kod nowej funkcji i wykonuje go przez exec.

    To jest moment autoprogramowania: powstaje funkcja,
    której nie pisaliśmy ręcznie.
    """
    primitive_names = list(PRIMITIVES.keys())

    # 1. Przeszukanie przestrzeni programów (kompozycji prymitywów)
    for depth in range(1, max_depth + 1):
        for seq in product(primitive_names, repeat=depth):
            if fits_examples(seq, examples):
                print(f"[MAG] Znaleziono sekwencję: {seq}")

                # 2. Budujemy ciało funkcji jako kod źródłowy
                body_expr = "x"
                for name in seq:
                    body_expr = f"{name}({body_expr})"

                src = f"""
def {func_name}(x):
    # Funkcja wygenerowana automatycznie przez silnik Maga
    # Sekwencja prymitywów: {seq}
    return {body_expr}
"""

                print("[MAG] Wygenerowany kod:")
                print(src)

                # 3. Wykonujemy kod i wstrzykujemy funkcję do globalnego namespace
                globals_dict = globals()
                exec(src, globals_dict)
                return globals_dict[func_name]

    raise ValueError("Nie znaleziono programu spełniającego wszystkie przykłady.")


# --- Demo dla kursu ---


if __name__ == "__main__":
    # Mówimy Magowi tylko tyle:
    # chcemy funkcję f, która:
    # 1 -> 11
    # 2 -> 14
    # 4 -> 20
    # (tak naprawdę f(x) = 3*x + 8, ale tego nie zdradzamy)
    training_examples: List[Example] = [
        (1, 11),
        (2, 14),
        (4, 20),
    ]

    spell = synthesize_spell_from_examples(training_examples, max_depth=4)

    print("== Test emergent_spell ==")
    for x, _ in training_examples:
        print(f"x = {x} -> emergent_spell(x) = {spell(x)}")

    # Możemy też użyć na nowym wejściu:
    print("x = 10 -> emergent_spell(10) =", spell(10))
