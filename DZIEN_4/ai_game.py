"""
narrative_map.py

A deterministic narrative decision-tree generator for tabletop RPGs,
built on top of ``networkx.DiGraph``.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class _ChoiceTemplate:
    """Internal helper describing a pair of contrasting choices."""
    left: str
    right: str


class NarrativeMap:
    """
    Generate an interactive narrative decision tree using a directed graph.

    The graph is rooted at a node named ``"start"``. Each node stores
    a short event description (2–3 sentences). Each directed edge stores
    decision metadata: ``choice`` label, ``risk`` and ``reward`` in range 1–10.

    Determinism is guaranteed by using a dedicated ``random.Random`` instance
    seeded in the constructor.

    :param themes: A list of narrative themes (e.g., "odkupienie", "tajemnica").
    :type themes: list[str]
    :param locations: A list of possible locations for events.
    :type locations: list[str]
    :param characters: A list of character archetypes/names for events.
    :type characters: list[str]
    :param seed: Seed for deterministic generation. If ``None``, a non-deterministic
        seed is used (not recommended when reproducibility is desired).
    :type seed: int | None

    :ivar themes: Stored themes used for description generation.
    :vartype themes: list[str]
    :ivar locations: Stored locations used for description generation.
    :vartype locations: list[str]
    :ivar characters: Stored characters used for description generation.
    :vartype characters: list[str]
    :ivar seed: Stored seed value.
    :vartype seed: int | None
    :ivar rng: Dedicated PRNG used throughout the instance.
    :vartype rng: random.Random
    :ivar graph: The underlying directed graph.
    :vartype graph: networkx.DiGraph

    **Node attributes**
        - ``description`` (str): event description (2–3 sentences)

    **Edge attributes**
        - ``choice`` (str): decision label
        - ``risk`` (int): 1..10
        - ``reward`` (int): 1..10

    Examples
    --------
    >>> nm = NarrativeMap(
    ...     themes=["odkupienie", "tajemnica"],
    ...     locations=["opuszczony zamek", "mglista dolina"],
    ...     characters=["Wędrowiec", "Cień"],
    ...     seed=42,
    ... )
    >>> nm.build_tree(depth=2)
    >>> path = ["start", "n0_L", "n0_LL"]
    >>> summary = nm.get_path_summary(path)
    >>> summary["n_steps"]
    2
    """

    def __init__(
        self,
        themes: List[str],
        locations: List[str],
        characters: List[str],
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the narrative map with inputs and an empty directed graph.

        The graph starts empty; after initialization a root node ``"start"``
        is inserted to act as the decision-tree root.

        :param themes: A list of narrative themes.
        :type themes: list[str]
        :param locations: A list of event locations.
        :type locations: list[str]
        :param characters: A list of characters (names/archetypes).
        :type characters: list[str]
        :param seed: Seed for deterministic PRNG; pass ``None`` for non-determinism.
        :type seed: int | None
        :raises ValueError: If any input list is empty.
        """
        if not themes or not locations or not characters:
            raise ValueError("themes, locations, and characters must be non-empty lists.")

        self.themes: List[str] = list(themes)
        self.locations: List[str] = list(locations)
        self.characters: List[str] = list(characters)
        self.seed: Optional[int] = seed

        self.rng: random.Random = random.Random(seed)
        self.graph: nx.DiGraph = nx.DiGraph()

        self.graph.add_node("start", description=self._generate_event_description("start"))

    def build_tree(self, depth: int = 3) -> None:
        """
        Build a deterministic decision tree rooted at node ``"start"``.

        Nodes are named with readable identifiers:
        - immediate children of ``"start"``: ``"n0_L"``, ``"n0_R"``
        - deeper nodes: ``"n0_LR"``, ``"n0_RLL"``, etc.

        Each non-leaf node has at least two outgoing edges (left/right),
        labeled with contrasting choice strings and edge attributes:
        ``choice``, ``risk`` and ``reward`` (each in 1..10).

        Cumulative risk and reward are not stored directly on nodes; they are
        computed by summing along a chosen path (see :meth:`get_path_summary`).

        :param depth: Depth of the tree in number of decision steps.
            A depth of 1 creates children of ``"start"`` only.
        :type depth: int
        :raises ValueError: If ``depth`` is less than 1.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")

        # Clear everything except the root "start" to allow rebuilding.
        root_desc = self.graph.nodes["start"].get("description", "")
        self.graph.clear()
        self.graph.add_node("start", description=root_desc or self._generate_event_description("start"))

        templates = self._choice_templates()

        def expand(parent_id: str, parent_code: str, remaining: int) -> None:
            if remaining <= 0:
                return

            tmpl = self.rng.choice(templates)

            left_child = self._make_child_id(parent_code, "L")
            right_child = self._make_child_id(parent_code, "R")

            if left_child not in self.graph:
                self.graph.add_node(left_child, description=self._generate_event_description(left_child))
            if right_child not in self.graph:
                self.graph.add_node(right_child, description=self._generate_event_description(right_child))

            # Deterministic risk/reward per edge.
            l_risk = self.rng.randint(1, 10)
            l_reward = self.rng.randint(1, 10)
            r_risk = self.rng.randint(1, 10)
            r_reward = self.rng.randint(1, 10)

            self.graph.add_edge(
                parent_id,
                left_child,
                choice=tmpl.left,
                risk=l_risk,
                reward=l_reward,
            )
            self.graph.add_edge(
                parent_id,
                right_child,
                choice=tmpl.right,
                risk=r_risk,
                reward=r_reward,
            )

            expand(left_child, left_child, remaining - 1)
            expand(right_child, right_child, remaining - 1)

        expand("start", "n0", depth)

    def get_path_summary(self, path: List[str]) -> Dict[str, object]:
        """
        Summarize a path from root to leaf (or any valid node sequence).

        The path must be a list of node identifiers present in the graph
        starting with ``"start"``. The method validates that each consecutive
        pair forms a directed edge.

        Returned summary includes event descriptions, decision labels for edges,
        and the total cumulative risk/reward (sum of edge attributes).

        :param path: A list of node ids from root to a destination node.
        :type path: list[str]
        :return: A dictionary with keys:
            - ``events`` (list[str]): event descriptions for each node
            - ``choices`` (list[str]): edge choice labels between nodes
            - ``total_risk`` (int): sum of edge risks
            - ``total_reward`` (int): sum of edge rewards
            - ``n_steps`` (int): number of transitions (len(path) - 1)
        :rtype: dict
        :raises ValueError: If the path is empty, does not start at ``"start"``,
            contains unknown nodes, or contains invalid transitions.
        """
        if not path:
            raise ValueError("path must be a non-empty list of node ids")
        if path[0] != "start":
            raise ValueError('path must start with node id "start"')
        for node_id in path:
            if node_id not in self.graph:
                raise ValueError(f"unknown node in path: {node_id}")

        events: List[str] = []
        choices: List[str] = []
        total_risk = 0
        total_reward = 0

        for i, node_id in enumerate(path):
            desc = self.graph.nodes[node_id].get("description")
            if not isinstance(desc, str) or not desc.strip():
                desc = self._generate_event_description(node_id)
                self.graph.nodes[node_id]["description"] = desc
            events.append(desc)

            if i == 0:
                continue

            u = path[i - 1]
            v = node_id
            if not self.graph.has_edge(u, v):
                raise ValueError(f"invalid transition in path: {u} -> {v}")

            edge_data = self.graph.get_edge_data(u, v) or {}
            choice = edge_data.get("choice")
            risk = edge_data.get("risk")
            reward = edge_data.get("reward")

            if not isinstance(choice, str):
                raise ValueError(f'missing or invalid "choice" on edge {u}->{v}')
            if not isinstance(risk, int):
                raise ValueError(f'missing or invalid "risk" on edge {u}->{v}')
            if not isinstance(reward, int):
                raise ValueError(f'missing or invalid "reward" on edge {u}->{v}')

            choices.append(choice)
            total_risk += risk
            total_reward += reward

        return {
            "events": events,
            "choices": choices,
            "total_risk": total_risk,
            "total_reward": total_reward,
            "n_steps": max(0, len(path) - 1),
        }

    def _generate_event_description(self, node_id: str) -> str:
        """
        Generate a short 2–3 sentence event description for a node.

        This method simulates a future call to a language model. The goal is to
        produce coherent mini-scenes that reuse the provided ``themes``,
        ``locations`` and ``characters`` in a deterministic manner.

        # TODO: call GPT here

        :param node_id: Node identifier for which to generate description.
        :type node_id: str
        :return: A 2–3 sentence event description.
        :rtype: str
        """
        # TODO: call GPT here

        theme = self._pick_from_list(self.themes, node_id, salt="theme")
        location = self._pick_from_list(self.locations, node_id, salt="loc")
        c1, c2 = self._pick_two_characters(node_id)

        # Small deterministic "beats" based on node id.
        beat = self._pick_from_list(
            [
                "odkrywa ślad, którego nie powinno tu być",
                "słyszy szept, który brzmi jak cudze wspomnienie",
                "widzi znak pozostawiony wbrew logice miejsca",
                "odnajduje przedmiot, który pasuje do jego historii zbyt dobrze",
                "czuje, że ktoś prowadzi go po niewidzialnej nici",
            ],
            node_id,
            salt="beat",
        )

        twist = self._pick_from_list(
            [
                "Cena jest ukryta w drobnym geście, nie w wielkiej bitwie.",
                "Prawda okazuje się lżejsza niż podejrzenia, ale bardziej niebezpieczna.",
                "To, co wygląda na ratunek, może być negocjacją z cieniem.",
                "Tu nie wygrywa się siłą, tylko wyborem, którego nie da się cofnąć.",
                "Wszystko zmierza ku odpowiedzi, która nie będzie wygodna.",
            ],
            node_id,
            salt="twist",
        )

        # Ensure 2–3 sentences.
        sentence1 = f"W {location} {c1} {beat}."
        sentence2 = f"Motywem tej sceny jest {theme}, a obecność {c2} zmienia znaczenie każdego słowa."
        if self.rng.random() < 0.55:
            sentence3 = twist
            return f"{sentence1} {sentence2} {sentence3}"
        return f"{sentence1} {sentence2}"

    def _choice_templates(self) -> List[_ChoiceTemplate]:
        """
        Return a curated set of contrasting decision label pairs.

        :return: Choice templates used for left/right branches.
        :rtype: list[_ChoiceTemplate]
        """
        return [
            _ChoiceTemplate("zaufaj", "zdrada"),
            _ChoiceTemplate("ucieknij", "staw czoła"),
            _ChoiceTemplate("negocjuj", "zaatakuj"),
            _ChoiceTemplate("ukryj prawdę", "wyznaj prawdę"),
            _ChoiceTemplate("podążaj za znakiem", "złam znak"),
            _ChoiceTemplate("ocal", "poświęć"),
            _ChoiceTemplate("wejdź w cień", "zapal światło"),
            _ChoiceTemplate("wysłuchaj", "zignoruj"),
        ]

    def _make_child_id(self, parent_code: str, branch: str) -> str:
        """
        Create a readable child node id from parent code and branch letter.

        The scheme is:
        - From root code "n0": children are "n0_L" and "n0_R"
        - From any node that already includes "_": append letters without extra "_"
          e.g., "n0_L" -> "n0_LL" / "n0_LR"

        :param parent_code: Parent node id or root code.
        :type parent_code: str
        :param branch: Branch letter, expected "L" or "R".
        :type branch: str
        :return: Child node id.
        :rtype: str
        """
        if branch not in {"L", "R"}:
            raise ValueError('branch must be "L" or "R"')

        if parent_code == "n0":
            return f"{parent_code}_{branch}"
        if "_".join(parent_code.split("_")[:1]) == "n0" and "_" not in parent_code:
            return f"{parent_code}_{branch}"
        if "_" in parent_code:
            return f"{parent_code}{branch}"
        return f"{parent_code}_{branch}"

    def _pick_from_list(self, items: List[str], node_id: str, salt: str) -> str:
        """
        Deterministically pick an element from a list using node id and salt.

        This method does not use the global ``random`` module; it relies on the
        instance PRNG but derives an index in a stable way.

        :param items: Non-empty list of strings.
        :type items: list[str]
        :param node_id: Node identifier.
        :type node_id: str
        :param salt: Additional salt to diversify picks per feature.
        :type salt: str
        :return: Selected item.
        :rtype: str
        """
        if not items:
            raise ValueError("items must be non-empty")
        key = f"{salt}:{node_id}"
        idx = self._stable_index(key, len(items))
        return items[idx]

    def _pick_two_characters(self, node_id: str) -> Tuple[str, str]:
        """
        Deterministically select two (possibly different) characters.

        :param node_id: Node identifier for selection context.
        :type node_id: str
        :return: Pair of characters.
        :rtype: tuple[str, str]
        """
        if len(self.characters) == 1:
            return self.characters[0], self.characters[0]

        first = self._pick_from_list(self.characters, node_id, salt="c1")
        second = self._pick_from_list(self.characters, node_id, salt="c2")

        if first == second:
            # Resolve collision deterministically by shifting index.
            idx = (self.characters.index(second) + 1) % len(self.characters)
            second = self.characters[idx]
        return first, second

    def _stable_index(self, key: str, modulo: int) -> int:
        """
        Create a stable pseudo-random index in range [0, modulo).

        The function mixes the key into an integer and reduces it modulo.
        It uses the instance RNG to vary mixing coefficients deterministically.

        :param key: String key to hash/mix.
        :type key: str
        :param modulo: Upper bound for modulo operation (must be > 0).
        :type modulo: int
        :return: Integer index in [0, modulo).
        :rtype: int
        """
        if modulo <= 0:
            raise ValueError("modulo must be > 0")

        # Simple deterministic rolling hash with RNG-derived multipliers.
        # Multipliers are deterministic given the instance seed and call order.
        a = 131 + self.rng.randint(0, 37)
        b = 17 + self.rng.randint(0, 23)

        h = 0
        for ch in key:
            h = (h * a + ord(ch) * b) & 0xFFFFFFFF
        return int(h % modulo)


if __name__ == "__main__":
    themes = ["odkupienie", "tajemnica"]
    locations = ["opuszczony zamek", "mglista dolina"]
    characters = ["Wędrowiec", "Cień"]

    nm = NarrativeMap(themes=themes, locations=locations, characters=characters, seed=42)
    nm.build_tree(depth=3)

    # Pick the deepest "leftmost" path: always take the first successor by sorted id.
    path = ["start"]
    current = "start"
    while True:
        succ = sorted(nm.graph.successors(current))
        if not succ:
            break
        current = succ[0]
        path.append(current)

    summary = nm.get_path_summary(path)

    print("=== PATH SUMMARY ===")
    print(f"Path: {' -> '.join(path)}")
    print(f"Steps: {summary['n_steps']}")
    print(f"Total risk: {summary['total_risk']}")
    print(f"Total reward: {summary['total_reward']}\n")

    print("Events:")
    for i, ev in enumerate(summary["events"], start=1):
        print(f"{i}. {ev}")

    print("\nChoices:")
    for i, ch in enumerate(summary["choices"], start=1):
        print(f"{i}. {ch}")

    print("\n=== EDGE ATTRIBUTES DEMO (first 10 edges) ===")
    for i, (u, v, data) in enumerate(nm.graph.edges(data=True)):
        if i >= 10:
            break
        choice = data.get("choice")
        risk = data.get("risk")
        reward = data.get("reward")
        print(f"{u} -> {v} | choice={choice!r}, risk={risk}, reward={reward}")
