import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_ResLSTM import Model_ResLSTM


def objfun(Soln):
    wave_Feat = Global_Vars.wave_Feat
    temp_Feat = Global_Vars.temp_Feat
    spact_Feat = Global_Vars.spact_Feat
    spat_Feat = Global_Vars.spat_Feat
    auto_Feat = Global_Vars.auto_Feat
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred_lstm = Model_ResLSTM(wave_Feat, temp_Feat, spact_Feat, spat_Feat, auto_Feat, Target, sol)
            Eval = evaluation(pred_lstm, Target)
            Fitn[i] = 1 / Eval[13]
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred_lstm = Model_ResLSTM(wave_Feat, temp_Feat, spact_Feat, spat_Feat, auto_Feat, Target, sol)
        Eval = evaluation(pred_lstm, Target)
        Fitn = 1 / Eval[13]
        return Fitn
