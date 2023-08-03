**pickle data format**:
- theta: trained model parameters
- yymmdd: date of experiments
- exn: experiment #n on that day

**experiments**:
- theta230718ex1: NLL cs273ahw3 original, the 1st entry is the const intercept, 
  the rest 104 entries represent energy values of nucliotide by order ACGT across
  26 positions at CPR binding sites [-74:-49]. The data is stored in a numpy array.
- theta230718ex2: the same as theta230718ex1 but with extra penalty on loss function
  depending on how far it is away from bin 5 (reference sequences).
- theta230718ex3: the same as theta230718ex1 but without rescaling the data
  before training the model.
- theta230718ex4: the same as theta230718ex3 but with extra penalty on loss function
  depending on how far it is away from bin 5 (reference sequences).
- theta230719 (crp and rnap): classic NLL loss.
- theta230720 (crp and rnap): same as 230719 but with zero intercept in the linear
  combination of energies from each site, which means the total binding energy.
- theta230724hinge (crp and rnap): same as 230719 but using hinge loss instead of logistic
- theta230726trm276param: model infers all parameters
- theta230727 (crp and rnap): same as 230719 but label revised, energy matrix function updated.
- theta230727joint272param: model infers all parameters (272 in total)
- theta230802joint272param: model infers all parameters (272 in total), new generated

**figures**:
- blue curve: surrogate loss
- red curve: actual error rate
