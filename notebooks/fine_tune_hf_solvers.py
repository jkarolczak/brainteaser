import sys

sys.path.append("/home/ubuntu/repositories/brainteaser/brainteaser")

from hf_solvers import FineTunedZephyr7BetaSolver
from structs import DataSet


if __name__ == "__main__":

    sp_solver = FineTunedZephyr7BetaSolver(model_name="argilla/notus-7b-v1")

    sp_train = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/SP-train.pkl")
    sp_eval = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/SP-eval.pkl")

    sp_solver.fine_tune(train_dataset=sp_train, eval_dataset=sp_eval, epochs=3)

    wp_solver = FineTunedZephyr7BetaSolver(model_name="argilla/notus-7b-v1")

    wp_train = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/WP-train.pkl")
    wp_eval = DataSet.from_file("/home/ubuntu/repositories/brainteaser/data/WP-eval.pkl")

    wp_solver.fine_tune(train_dataset=wp_train, eval_dataset=wp_eval, epochs=3)
