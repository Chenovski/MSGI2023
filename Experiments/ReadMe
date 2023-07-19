pickle data format:
- theta: trained model parameters
- yymmdd: date of experiments
- exn: experiment #n on that day

experiments:
- theta230718ex1: NLL cs273ahw3 original, the 1st entry is the const intercept, 
  the rest 104 entries represent energy values of nucliotide by order ACGT across
  26 positions at CPR binding sites [-74:-49]. The data is stored in a numpy array.
- theta230718ex2: the same as theta230718ex1 but with extra penalty on loss function
  depending on how far it is away from bin 5 (reference sequences).
- theta230718ex3: the same as theta230718ex1 but without rescaling the data
  before training the model.
- theta230718ex4: the same as theta230718ex3 but with extra penalty on loss function
  depending on how far it is away from bin 5 (reference sequences).

figures:
- blue curve: surrogate loss
- red curve: actual error rate
