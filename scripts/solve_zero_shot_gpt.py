from solvers import ZeroShotGPT

from structs import DataSet

if __name__ == "__main__":
    solver = ZeroShotGPT()
    sp = DataSet.from_file("./data/SP-train.pkl")

    solver.solve(sp[0])
