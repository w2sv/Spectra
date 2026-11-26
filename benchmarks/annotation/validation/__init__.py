"""Three numerical experiments supporting one maintained annotation workflow."""

from enum import StrEnum
from typing import Literal


class ValidationStrategy(StrEnum):
    ACOUSTIC_ONSET = "acoustic-onset"
    GUIDED_TEMPLATE_MATCHING = "guided-template-matching"
    AUTOMATIC_TEMPLATE_MATCHING = "automatic-template-matching"


def validate(
    strategy: ValidationStrategy | Literal["all"] = ValidationStrategy.AUTOMATIC_TEMPLATE_MATCHING,
) -> dict[str, object]:
    from benchmarks.annotation.validation.acoustic_onset import validate_acoustic_onset
    from benchmarks.annotation.validation.automatic_template_matching import validate_automatic_template_matching
    from benchmarks.annotation.validation.guided_template_matching import validate_guided_template_matching

    implementations = {
        ValidationStrategy.ACOUSTIC_ONSET: validate_acoustic_onset,
        ValidationStrategy.GUIDED_TEMPLATE_MATCHING: validate_guided_template_matching,
        ValidationStrategy.AUTOMATIC_TEMPLATE_MATCHING: validate_automatic_template_matching,
    }
    if strategy == "all":
        return {"strategies": {name.value: run() for name, run in implementations.items()}}
    return implementations[strategy]()
