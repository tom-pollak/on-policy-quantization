"""
Minimal subclasses of GKDConfig and GKDTrainer to add min_new_tokens support.

The upstream trl GKDConfig doesn't expose min_new_tokens, so we subclass both
the config (to accept the arg) and the trainer (to pass it to generation_kwargs).
"""

from dataclasses import dataclass, field

from trl.experimental.gkd import (
    GKDConfig as BaseGKDConfig,
    GKDTrainer as BaseGKDTrainer,
)

from eval import compute_perplexity


@dataclass
class GKDConfig(BaseGKDConfig):
    min_new_tokens: int = 1
    perplexity_dataset: str | None = field(
        default=None,
        metadata={"help": "Dataset for perplexity evaluation (wikitext, c4)"},
    )


class GKDTrainer(BaseGKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config.min_new_tokens = self.args.min_new_tokens
        self.generation_kwargs["min_new_tokens"] = self.args.min_new_tokens

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = {}

        if self.args.perplexity_dataset is not None:
            self.model.eval()
            ppl = compute_perplexity(
                self.model,
                self.processing_class,
                dataset=self.args.perplexity_dataset,
            )
            if ppl is not None:
                metrics[
                    f"{metric_key_prefix}/perplexity_{self.args.perplexity_dataset}"
                ] = ppl
            self.model.train()

        return metrics
