from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .architecture import Architecture
from .controller import Decision, ExhaustiveController, RandomController, TabularPolicyController
from .evaluator import EvaluationResult, Evaluator
from .yolo_adapter import YoloArchitectureAdapter


@dataclass
class SearchRecord:
    episode: int
    architecture: Architecture
    evaluation: EvaluationResult

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "architecture": self.architecture.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


class ArchitectureSearchRunner:
    def __init__(
        self,
        adapter: YoloArchitectureAdapter,
        controller: Union[TabularPolicyController, RandomController, ExhaustiveController],
        evaluator: Evaluator,
        out_dir: Path,
    ) -> None:
        self.adapter = adapter
        self.controller = controller
        self.evaluator = evaluator
        self.out_dir = out_dir

    def run(self, episodes: int) -> SearchRecord:
        if episodes < 1:
            raise ValueError("episodes must be positive")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.out_dir / "history.jsonl"
        best: Optional[SearchRecord] = None
        records: List[SearchRecord] = []

        with history_path.open("w", encoding="utf-8") as history_file:
            for episode in range(1, episodes + 1):
                architecture, decisions = self._sample_architecture()
                evaluation = self.evaluator.evaluate(architecture)
                self.controller.update(decisions, evaluation.reward)
                record = SearchRecord(episode, architecture, evaluation)
                records.append(record)
                history_file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                if best is None or evaluation.reward > best.evaluation.reward:
                    best = record

        assert best is not None
        self._write_best(best)
        return best

    def _sample_architecture(self) -> tuple[Architecture, List[Decision]]:
        state = self.adapter.search_space.initial_state()
        architecture = self.adapter.empty_architecture()
        decisions: List[Decision] = []

        while not architecture.stopped:
            actions = self.adapter.search_space.valid_actions(state)
            action, decision = self.controller.select(state, actions)
            decisions.append(decision)
            self.adapter.apply_action(architecture, action)
            state = self.adapter.search_space.transition(state, action)
            if state["position"] >= self.adapter.search_space.max_blocks:
                architecture.stopped = True

        return architecture, decisions

    def _write_best(self, best: SearchRecord) -> None:
        arch_path = self.out_dir / "best_architecture.json"
        yolo_path = self.out_dir / "best_yolo.yaml"
        with arch_path.open("w", encoding="utf-8") as arch_file:
            json.dump(best.to_dict(), arch_file, indent=2, sort_keys=True)
            arch_file.write("\n")
        with yolo_path.open("w", encoding="utf-8") as yolo_file:
            yolo_file.write(self.adapter.to_yolo_yaml(best.architecture))
            yolo_file.write("\n")
