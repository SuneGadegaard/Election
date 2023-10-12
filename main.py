import pyomo.environ as pyomo
import json as js


def readData(fileName: str) -> dict:
    data = dict()
    with open(fileName) as d:
        data = js.load(d)
    return data


def buildModel(data: dict, weights=[1, 1, 1]) -> pyomo.ConcreteModel():
    model = pyomo.ConcreteModel()
    model.parties = data['parties']
    model.regions = data['regions']
    regionRange = range(len(model.regions))
    partyRange = range(len(model.parties))
    model.totalVotes = sum(data['votes'][j][p] for j in regionRange for p in partyRange)
    model.x = pyomo.Var(model.regions, model.parties, within=pyomo.NonNegativeIntegers)
    model.lamda = pyomo.Var(model.parties, within=pyomo.NonNegativeReals)
    model.mu = pyomo.Var(model.regions, model.parties, within=pyomo.NonNegativeReals)
    model.zeta = pyomo.Var(model.regions, within=pyomo.NonNegativeReals)
    model.obj = pyomo.Objective(expr=weights[0] * sum(model.lamda[p] for p in model.parties)
                                + weights[1] * sum(model.mu[j, p] for j in model.regions for p in model.parties)
                                + weights[2] * sum(model.zeta[j] for j in model.regions)
                                )

    model.totalSeats = pyomo.Constraint(
        expr=sum(model.x[j, p] for j in model.regions for p in model.parties) == data['numberOfSeats']
    )
    model.regionMinSeats = pyomo.ConstraintList()
    for j in model.regions:
        model.regionMinSeats.add(
            expr=sum(model.x[j, p] for p in model.parties) >= data['minNumSeatsPerRegion']
        )
    model.lambdaCsts = pyomo.ConstraintList()
    for pp, p in enumerate(model.parties):
        model.lambdaCsts.add(
            expr=model.lamda[p] >= sum(model.x[j, p] for j in model.regions) -
                 data['numberOfSeats']*sum(data['votes'][j][pp] for j in regionRange) / model.totalVotes
        )
        model.lambdaCsts.add(
            expr=model.lamda[p] >= data['numberOfSeats']*sum(data['votes'][j][pp] for j in regionRange) / model.totalVotes - sum(model.x[j, p] for j in model.regions)
        )

    model.muCsts = pyomo.ConstraintList()
    for jj, j in enumerate(model.regions):
        for pp, p in enumerate(model.parties):
            model.muCsts.add(
                expr=model.mu[j, p] >= model.x[j, p] - ( sum(data['votes'][l][pp] for l in regionRange)/model.totalVotes )*sum(model.x[j, q] for q in model.parties)
            )
            model.muCsts.add(
                expr=model.mu[j, p] >= ( sum(data['votes'][l][pp] for l in regionRange)/model.totalVotes )*sum(model.x[j, q] for q in model.parties) - model.x[j, p]
            )
    model.zetaCsts = pyomo.ConstraintList()
    for jj, j in enumerate(model.regions):
        model.zetaCsts.add(
            expr=model.zeta[j] >= sum(model.x[j, p] for p in model.parties) - (data['registered'][jj]/sum(data['registered']))*data['numberOfSeats']
        )
        model.zetaCsts.add(
            expr=model.zeta[j] >= (data['registered'][jj]/sum(data['registered']))*data['numberOfSeats'] - sum(model.x[j, p] for p in model.parties)
        )
    return model


def solveModel(model: pyomo.ConcreteModel()):
    solver = pyomo.SolverFactory('gurobi')
    solver.solve(model, tee=True)


def printSolution(model: pyomo.ConcreteModel()):
    for j in model.regions:
        print([max(pyomo.value(model.x[j, p]),0) for p in model.parties])


if __name__ == '__main__':
    fileName = 'electionData-3p-5d.json'
    data = readData(fileName)
    model = buildModel(data)
    solveModel(model)
    printSolution(model)
