"""
EXTREME REFLECTION / INTROSPECTION DEMO
---------------------------------------

Pokazuje:
1. Podglądanie całego stanu Pythona przez globals() / locals().
2. Dynamiczne tworzenie klasy z type().
3. Dynamiczny import + modyfikacja modułu w locie.
4. Reload klasy po modyfikacji.
5. Monkey-patching: obiekt zmienia swoją logikę w runtime.
6. Samoopisujący się obiekt.

To jest poziom MAG/MĘDRZEC — absolutne kulisy Pythona.
"""

import importlib
import os
import time
from types import ModuleType


# ====================================================
# 1. INSPEKCJA STANU W PAMIĘCI
# ====================================================

def show_globals_info():
    print("=== GLOBAL SCOPE ===")
    for name in list(globals().keys()):
        if not name.startswith("__"):
            print(" •", name)


def show_locals_info():
    print("=== LOCALS SCOPE ===")
    for name in list(locals().keys()):
        print(" •", name)


# ====================================================
# 2. DYNAMICZNA KREACJA KLASY
# ====================================================

def build_class(name: str):
    """
    Tworzymy klasę w runtime za pomocą type().
    """
    def talk(self):
        print(f"[DYNAMIC CLASS] I am instance of {name}")

    return type(name, (object,), {"talk": talk})


# ====================================================
# 3. TWORZENIE MODUŁU W TRAKCIE PROGRAMU
# ====================================================

def create_temp_module():
    """
    Tworzymy moduł plikowy, który za chwilę załadujemy dynamicznie.
    """

    code = """
print("[module_x] Loaded module_x")

def greet():
    print("[module_x] Hello from version 1!")
"""
    with open("module_x.py", "w", encoding="utf-8") as f:
        f.write(code)


def modify_temp_module():
    """
    Modyfikujemy moduł po czasie.
    """
    code = """
print("[module_x] Reloaded module_x – VERSION 2!")

def greet():
    print("[module_x] Hello from version 2 — reloaded!")
"""
    with open("module_x.py", "w", encoding="utf-8") as f:
        f.write(code)


# ====================================================
# 4. URUCHOMIENIE DEMO
# ====================================================

if __name__ == "__main__":

    print("\n=== 1) GLOBALS / LOCALS ===")
    show_globals_info()
    print()
    show_locals_info()

    print("\n=== 2) DYNAMIC CLASS CREATION ===")
    DynamicHero = build_class("DynamicHero")
    hero = DynamicHero()
    hero.talk()

    print("\n=== 3) DYNAMIC MODULE LOAD ===")
    create_temp_module()

    # Dynamiczny import
    module_x: ModuleType = importlib.import_module("module_x")
    module_x.greet()

    print("\nMODYFIKUJEMY moduł za 2 sekundy...")
    time.sleep(2)
    modify_temp_module()

    print("\n=== 4) MODULE RELOAD ===")
    module_x = importlib.reload(module_x)
    module_x.greet()

    print("\n=== 5) MONKEY PATCHING – zmiana logiki obiektu w runtime ===")

    class Warrior:
        def hit(self):
            print("Hit: standard attack.")

    w = Warrior()
    w.hit()

    # Monkey patch
    def new_attack(self):
        print("Hit: LIGHTNING ATTACK!")

    Warrior.hit = new_attack

    print("Po patchu:")
    w.hit()  # obiekt zmienia zachowanie bez reinstancji

    print("\n=== 6) OBIEKT SAMOOPISUJĄCY SIĘ ===")

    class SelfAware:
        def who_am_i(self):
            print("I am:", self.__class__.__name__)
            print("My methods:", list(self.__class__.__dict__.keys()))
            print("My memory:", self.__dict__)

    sa = SelfAware()
    sa.new_attr = 123
    sa.who_am_i()
