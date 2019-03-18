#####  Forward Select from any Data Set using Linear Models
import warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf
import statsmodels.api as sm
import time
from scipy import stats
def Forward_Select(data, response, modeltype, metric):
    """Select Variables using forward selection before building a Linear model.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response
    response: string, name of response column in data
    model_type: It accepts both "Regression" and "Classification" type problems.
    metric: the criteria improving which the variable is Selected.
          The metric must be a known metric among all Statsmodels' model metrics.

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model with an intercept
           selected by forward selection.
           evaluated by adjusted R-squared or AIC or BIC or whatever
    selected: variables that are selected by this algorithm.
    """
    ############################################################################
    #####   CAUTION CAUTION: IF you have Scipy 1.0 version you have to do this
    ##### This is a dumb workaround until Scipy 1.0 is patched - I should not have
    ### upgraded from scipy 0.19 to scipy 1.0 = full of bugs!![]. If you DONT
    #### have this statement then your glm.summary statement will give an ERROR
    stats.chisqprob = lambda chisq, data: stats.chi2.sf(chisq, data)
    #### For those who have Scipy 0.19 or older, you can comment out above line.
    ############################################################################
    start_time = time.time()
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    maxiter = 1000
    if metric == 'rsquared' or metric == 'rsquared_adj':
        current_score, best_new_score = 0.0, 0.0
    else:
        current_score, best_new_score = np.inf, np.inf
    iterations = 1
    if data[response].dtype == object:
        response_char = 'C(' + response + ')'
        data[response], factors = factorize_class(data[response])
    else:
        response_char = response
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            #print('Variable considered: %s' %candidate)
            if data[candidate].dtype == object:
                ### If it is a categorical variable, encode it this way
                #### In smf formula string notation, you don't have to add 1. it adds it automatically.
                if selected == []:
                    formula = "{} ~ {}".format(response_char,
                                                'C('+candidate+')')
                else:
                    formula = "{} ~ {} + {}".format(response_char,
                                            ' + '.join(selected), 'C('+candidate+')')
            else:
                formula = "{} ~ {}".format(response_char,
                                           ' + '.join(selected + [candidate]))
            if modeltype == 'Regression':
                model = smf.ols(formula, data).fit(max_iter=maxiter, disp=0)
            else:
                if len(data[response].value_counts()) > 2:
                    try:
                        model = smf.mnlogit(formula=formula, data=data).fit(max_iter=maxiter, disp=0)
                    except:
                        model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
                                        max_iter=maxiter, disp=0)
                else:
                    try:
                        model = smf.logit(formula=formula, data=data).fit(max_iter=maxiter, disp=0)
                    except:
                        model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
                                                    max_iter=maxiter, disp=0)
            try:
                score = eval('model.'+metric)
            except:
                if modeltype == 'Regression':
                    metric = 'aic'
                    print('Metric not recognized. Choosing default = %s' %metric)
                else:
                    metric = 'aic'
                    print('Metric not recognized. Choosing default = %s' %metric)
                score = eval('model.'+metric)
            iterations += 1
            scores_with_candidates.append((score, candidate))
        if metric == 'rsquared' or metric == 'rsquared_adj':
            scores_with_candidates.sort(reverse=False)
        else:
            scores_with_candidates.sort(reverse=False)
        best_new_score, best_candidate = scores_with_candidates.pop()
        if metric == 'rsquared' or metric == 'rsquared_adj':
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        else:
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
    tempform = []
    print('Time taken for %d iterations (minutes): %0.2f' %(iterations, (time.time()-start_time)/60))
    for eachcol in selected:
        if tempform == []:
            if data[eachcol].dtype == object:
                ### If it is a categorical variable, encode it this way
                tempform = 'C('+eachcol+')'
            else:
                tempform = eachcol
        else:
            if data[eachcol].dtype == object:
                ### If it is a categorical variable, encode it this way
                tempform = "{} + {}".format(tempform, 'C()'+eachcol+')')
            else:
                tempform = "{} + {}".format(tempform, eachcol)
    ### when all is done, put the formula together ####
    formula = "{} ~ {} ".format(response_char, tempform)
    if modeltype == 'Regression':
        model = smf.ols(formula, data).fit(max_iter=maxiter, disp=0)
    else:
        if len(data[response].value_counts()) > 2:
            try:
                model = smf.mnlogit(formula=formula, data=data).fit(max_iter=maxiter, disp=0)
            except:
                model = smf.glm(formula=formula, data=data, family=sm.families.Gamma()).fit(
                                        max_iter=maxiter, disp=0)
        else:
            try:
                model = smf.logit(formula, data).fit(max_iter=maxiter, disp=0)
            except:
                model = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit(
                                        max_iter=maxiter, disp=0)
    print('Score = %0.2f, Number Selected = %d\nmodel formula: %s' %(score,
                                    len(selected),formula))
    print('Time taken for Final Model (minutes): %0.2f' %((time.time()-start_time)/60))
    print(model.summary())
    return model, selected

x_train = train[0:6524]
x_validation = train[6524:len(train)]




Usage:

Forward_Select(x_train,'Item_Outlet_Sales','Regression','rsquared_adj')

