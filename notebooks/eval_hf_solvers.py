import os
import sys
new_path = '/home/ubuntu/repositories/brainteaser/brainteaser'
sys.path.append(new_path)
import wandb
os.environ["WANDB_SILENT"] = "true"

from hf_solvers import InContextZephyr
from structs import DataSet

if __name__ == "__main__":
    for i in ["zero", "one", "two", "three"]:
        solver = InContextZephyr(context=InContextZephyr.Context.SENTENCE, model_name="argilla/notus-7b-v1") # TODO: specify model name
        if i != "zero":
            solver.load_fine_tuned(f"/home/ubuntu/repositories/brainteaser/sp_{i}_shot_notus") # TODO: change this to the correct path

        wandb.init(
            project="brainteasers",
            name=f"InContext_{i}_shot_{solver.model_name}",
            config={
                "solver": f"{i}_shot_{solver.model_name}", 
            }
        )

        sp = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/SP-eval.pkl")

        sp_answers = solver.solve(sp)
        sp_are_answers_correct = [instance.is_choice_correct(answer) for instance, answer in zip(sp, sp_answers)]
        sp_accuracy = sum(sp_are_answers_correct) / len(sp_are_answers_correct)

        print(f"Accuracy on the Sentence Puzzle dataset: {sp_accuracy: .4f}")

        solver = InContextZephyr(context=InContextZephyr.Context.WORD,  model_name="argilla/notus-7b-v1") # TODO: specify model name
        if i != "zero":
            solver.load_fine_tuned(f"/home/ubuntu/repositories/brainteaser/wp_{i}_shot_notus") # TODO: change this to the correct path

        wp = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/WP-eval.pkl")

        wp_answers = solver.solve(wp)
        wp_are_answers_correct = [instance.is_choice_correct(answer) for instance, answer in zip(wp, wp_answers)]
        wp_accuracy = sum(wp_are_answers_correct) / len(wp_are_answers_correct)

        print(f"Accuracy on the Word Puzzle dataset: {wp_accuracy: .4f}")

        total_cardinality = (len(sp_answers) + len(wp_answers))
        total_accuracy = (sp_accuracy * (len(sp_answers) / total_cardinality) +
                        wp_accuracy * (len(wp_answers) / total_cardinality))
        
        wandb.log(
            {
                "accuracy/overall": total_accuracy,
                "accuracy/sp": sp_accuracy,
                "accuracy/wp": wp_accuracy,
            }
        )

        wandb.finish(quiet=True)