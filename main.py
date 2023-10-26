import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from classes import *
import uvicorn

# uvicorn main:app --reload
# uvicorn main:app --host 0.0.0.0 --port 80

app = FastAPI()

loan_predictor = joblib.load("loan_classifier.joblib")
grade_predictor = joblib.load("grade_classifier.joblib")
sub_grade_predictor = joblib.load("sub_grade_classifier.joblib")
interest_rate_predictor = joblib.load("int_rate_regressor.joblib")


@app.get("/")
def home():
    return {"text": "Welcome to Lending Club loan predictions application"}


@app.post("/loan_prediction")
async def create_application(loan_pred: LoanPrediction):
    loan_df = pd.DataFrame()

    loan_df["amount_requested"] = [loan_pred.amount_requested]
    loan_df["risk_score"] = [loan_pred.risk_score]
    loan_df["debt_to_income_ratio"] = [loan_pred.debt_to_income_ratio]
    loan_df["employment_length"] = [loan_pred.employment_length]
    loan_df["year"] = [loan_pred.year]
    loan_df["risk_times_dti"] = [loan_pred.risk_times_dti]

    prediction = loan_predictor.predict(loan_df)
    if prediction[0] == 0:
        prediction = "Your loan application is rejected"
    else:
        prediction = "Your loan application is approved"

    return {"prediction": prediction}


@app.post("/grade_prediction")
async def create_application(grade_pred: GradePrediction):
    grade_df = pd.DataFrame()

    if grade_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(
            status_code=404,
            detail='This home ownership status is not found. Please choose from "MORTGAGE", "OWN", "RENT", "ANY", "NONE", "OTHER"',
        )

    if grade_pred.verification_status_joint not in verification_status_joint_dict:
        raise HTTPException(
            status_code=404,
            detail='Please select a valid verification status joint from : "Not Verified", "Verified", "Source Verified"',
        )

    if grade_pred.initial_list_status_dict not in initial_list_status_dict:
        raise HTTPException(
            status_code=404,
            detail='This initial list status is not found. Please choose from "w", "f"',
        )

    if grade_pred.disbursement_method not in disbursement_method_dict:
        raise HTTPException(
            status_code=404,
            detail='Please select a valid verification status joint from : "Cash", "DirectPay"',
        )

    grade_df["loan_amnt"] = [grade_pred.loan_amnt]
    grade_df["funded_amnt"] = [grade_pred.funded_amnt]
    grade_df["funded_amnt_inv"] = [grade_pred.funded_amnt_inv]
    grade_df["term"] = [grade_pred.term]
    grade_df["installment"] = [grade_pred.installment]
    grade_df["emp_length"] = [grade_pred.emp_length]
    grade_df["home_ownership"] = [grade_pred.home_ownership]
    grade_df["annual_inc"] = [grade_pred.annual_inc]
    grade_df["dti"] = [grade_pred.dti]
    grade_df["delinq_2yrs"] = [grade_pred.delinq_2yrs]
    grade_df["inq_last_6mths"] = [grade_pred.inq_last_6mths]
    grade_df["mths_since_last_delinq"] = [grade_pred.mths_since_last_delinq]
    grade_df["mths_since_last_record"] = [grade_pred.mths_since_last_record]
    grade_df["open_acc"] = [grade_pred.open_acc]
    grade_df["pub_rec"] = [grade_pred.pub_rec]
    grade_df["revol_bal"] = [grade_pred.revol_bal]
    grade_df["revol_util"] = [grade_pred.revol_util]
    grade_df["total_acc"] = [grade_pred.total_acc]
    grade_df["initial_list_status"] = [grade_pred.initial_list_status]
    grade_df["out_prncp"] = [grade_pred.out_prncp]
    grade_df["out_prncp_inv"] = [grade_pred.out_prncp_inv]
    grade_df["total_pymnt"] = [grade_pred.total_pymnt]
    grade_df["total_pymnt_inv"] = [grade_pred.total_pymnt_inv]
    grade_df["total_rec_prncp"] = [grade_pred.total_rec_prncp]
    grade_df["total_rec_int"] = [grade_pred.total_rec_int]
    grade_df["total_rec_late_fee"] = [grade_pred.total_rec_late_fee]
    grade_df["recoveries"] = [grade_pred.recoveries]
    grade_df["collection_recovery_fee"] = [grade_pred.collection_recovery_fee]
    grade_df["last_pymnt_amnt"] = [grade_pred.last_pymnt_amnt]
    grade_df["mths_since_last_major_derog"] = [grade_pred.mths_since_last_major_derog]
    grade_df["verification_status_joint"] = [grade_pred.verification_status_joint]
    grade_df["tot_coll_amt"] = [grade_pred.tot_coll_amt]
    grade_df["tot_cur_bal"] = [grade_pred.tot_cur_bal]
    grade_df["open_acc_6m"] = [grade_pred.open_acc_6m]
    grade_df["open_act_il"] = [grade_pred.open_act_il]
    grade_df["open_il_12m"] = [grade_pred.open_il_12m]
    grade_df["open_il_24m"] = [grade_pred.open_il_24m]
    grade_df["mths_since_rcnt_il"] = [grade_pred.mths_since_rcnt_il]
    grade_df["total_bal_il"] = [grade_pred.total_bal_il]
    grade_df["il_util"] = [grade_pred.il_util]
    grade_df["open_rv_12m"] = [grade_pred.open_rv_12m]
    grade_df["open_rv_24m"] = [grade_pred.open_rv_24m]
    grade_df["max_bal_bc"] = [grade_pred.max_bal_bc]
    grade_df["all_util"] = [grade_pred.all_util]
    grade_df["total_rev_hi_lim"] = [grade_pred.total_rev_hi_lim]
    grade_df["inq_fi"] = [grade_pred.inq_fi]
    grade_df["total_cu_tl"] = [grade_pred.total_cu_tl]
    grade_df["inq_last_12m"] = [grade_pred.inq_last_12m]
    grade_df["acc_open_past_24mths"] = [grade_pred.acc_open_past_24mths]
    grade_df["avg_cur_bal"] = [grade_pred.avg_cur_bal]
    grade_df["bc_open_to_buy"] = [grade_pred.bc_open_to_buy]
    grade_df["bc_util"] = [grade_pred.bc_util]
    grade_df["mo_sin_old_il_acct"] = [grade_pred.mo_sin_old_il_acct]
    grade_df["mo_sin_old_rev_tl_op"] = [grade_pred.mo_sin_old_rev_tl_op]
    grade_df["mo_sin_rcnt_rev_tl_op"] = [grade_pred.mo_sin_rcnt_rev_tl_op]
    grade_df["mo_sin_rcnt_tl"] = [grade_pred.mo_sin_rcnt_tl]
    grade_df["mort_acc"] = [grade_pred.mort_acc]
    grade_df["mths_since_recent_bc"] = [grade_pred.mths_since_recent_bc]
    grade_df["mths_since_recent_bc_dlq"] = [grade_pred.mths_since_recent_bc_dlq]
    grade_df["mths_since_recent_inq"] = [grade_pred.mths_since_recent_inq]
    grade_df["mths_since_recent_revol_delinq"] = [
        grade_pred.mths_since_recent_revol_delinq
    ]
    grade_df["num_accts_ever_120_pd"] = [grade_pred.num_accts_ever_120_pd]
    grade_df["num_actv_bc_tl"] = [grade_pred.num_actv_bc_tl]
    grade_df["num_actv_rev_tl"] = [grade_pred.num_actv_rev_tl]
    grade_df["num_bc_sats"] = [grade_pred.num_bc_sats]
    grade_df["num_bc_tl"] = [grade_pred.num_bc_tl]
    grade_df["num_il_tl"] = [grade_pred.num_il_tl]
    grade_df["num_op_rev_tl"] = [grade_pred.num_op_rev_tl]
    grade_df["num_rev_accts"] = [grade_pred.num_rev_accts]
    grade_df["num_rev_tl_bal_gt_0"] = [grade_pred.num_rev_tl_bal_gt_0]
    grade_df["num_sats"] = [grade_pred.num_sats]
    grade_df["num_tl_90g_dpd_24m"] = [grade_pred.num_tl_90g_dpd_24m]
    grade_df["num_tl_op_past_12m"] = [grade_pred.num_tl_op_past_12m]
    grade_df["pct_tl_nvr_dlq"] = [grade_pred.pct_tl_nvr_dlq]
    grade_df["percent_bc_gt_75"] = [grade_pred.percent_bc_gt_75]
    grade_df["pub_rec_bankruptcies"] = [grade_pred.pub_rec_bankruptcies]
    grade_df["tot_hi_cred_lim"] = [grade_pred.tot_hi_cred_lim]
    grade_df["total_bal_ex_mort"] = [grade_pred.total_bal_ex_mort]
    grade_df["total_bc_limit"] = [grade_pred.total_bc_limit]
    grade_df["total_il_high_credit_limit"] = [grade_pred.total_il_high_credit_limit]
    grade_df["disbursement_method"] = [grade_pred.disbursement_method]
    grade_df["Year_issue_d"] = [grade_pred.Year_issue_d]
    grade_df["cos_Month_issue_d"] = [grade_pred.cos_Month_issue_d]
    grade_df["sin_Month_issue_d"] = [grade_pred.sin_Month_issue_d]
    grade_df["log_loan_amnt"] = [grade_pred.log_loan_amnt]
    grade_df["risk_score"] = [grade_pred.risk_score]

    prediction = sub_grade_predictor.predict(grade_df)
    if prediction[0] == 0:
        prediction = "You are likely to get grade 'A'"
    elif prediction[0] == 1:
        prediction = "You are likely to get grade 'B'"
    elif prediction[0] == 2:
        prediction = "You are likely to get grade 'C'"
    elif prediction[0] == 3:
        prediction = "You are likely to get grade 'D'"
    elif prediction[0] == 4:
        prediction = "You are likely to get grade 'E'"
    elif prediction[0] == 5:
        prediction = "You are likely to get grade 'F'"
    else:
        prediction = "You are likely to get grade 'G'"

    return {"prediction": prediction}


@app.post("/sub_grade_prediction")
async def create_application(sub_grade_pred: GradePrediction):
    sub_df = pd.DataFrame()

    if sub_grade_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(
            status_code=404,
            detail='This home ownership status is not found. Please choose from "MORTGAGE", "OWN", "RENT", "ANY", "NONE", "OTHER"',
        )

    if sub_grade_pred.verification_status_joint not in verification_status_joint_dict:
        raise HTTPException(
            status_code=404,
            detail='Please select a valid verification status joint from : "Not Verified", "Verified", "Source Verified"',
        )

    if sub_grade_pred.initial_list_status_dict not in initial_list_status_dict:
        raise HTTPException(
            status_code=404,
            detail='This initial list status is not found. Please choose from "w", "f"',
        )

    if sub_grade_pred.disbursement_method not in disbursement_method_dict:
        raise HTTPException(
            status_code=404,
            detail='Please select a valid verification status joint from : "Cash", "DirectPay"',
        )

    sub_df["loan_amnt"] = [sub_grade_pred.loan_amnt]
    sub_df["funded_amnt"] = [sub_grade_pred.funded_amnt]
    sub_df["funded_amnt_inv"] = [sub_grade_pred.funded_amnt_inv]
    sub_df["term"] = [sub_grade_pred.term]
    sub_df["installment"] = [sub_grade_pred.installment]
    sub_df["emp_length"] = [sub_grade_pred.emp_length]
    sub_df["home_ownership"] = [sub_grade_pred.home_ownership]
    sub_df["annual_inc"] = [sub_grade_pred.annual_inc]
    sub_df["dti"] = [sub_grade_pred.dti]
    sub_df["delinq_2yrs"] = [sub_grade_pred.delinq_2yrs]
    sub_df["inq_last_6mths"] = [sub_grade_pred.inq_last_6mths]
    sub_df["mths_since_last_delinq"] = [sub_grade_pred.mths_since_last_delinq]
    sub_df["mths_since_last_record"] = [sub_grade_pred.mths_since_last_record]
    sub_df["open_acc"] = [sub_grade_pred.open_acc]
    sub_df["pub_rec"] = [sub_grade_pred.pub_rec]
    sub_df["revol_bal"] = [sub_grade_pred.revol_bal]
    sub_df["revol_util"] = [sub_grade_pred.revol_util]
    sub_df["total_acc"] = [sub_grade_pred.total_acc]
    sub_df["initial_list_status"] = [sub_grade_pred.initial_list_status]
    sub_df["out_prncp"] = [sub_grade_pred.out_prncp]
    sub_df["out_prncp_inv"] = [sub_grade_pred.out_prncp_inv]
    sub_df["total_pymnt"] = [sub_grade_pred.total_pymnt]
    sub_df["total_pymnt_inv"] = [sub_grade_pred.total_pymnt_inv]
    sub_df["total_rec_prncp"] = [sub_grade_pred.total_rec_prncp]
    sub_df["total_rec_int"] = [sub_grade_pred.total_rec_int]
    sub_df["total_rec_late_fee"] = [sub_grade_pred.total_rec_late_fee]
    sub_df["recoveries"] = [sub_grade_pred.recoveries]
    sub_df["collection_recovery_fee"] = [sub_grade_pred.collection_recovery_fee]
    sub_df["last_pymnt_amnt"] = [sub_grade_pred.last_pymnt_amnt]
    sub_df["mths_since_last_major_derog"] = [sub_grade_pred.mths_since_last_major_derog]
    sub_df["verification_status_joint"] = [sub_grade_pred.verification_status_joint]
    sub_df["tot_coll_amt"] = [sub_grade_pred.tot_coll_amt]
    sub_df["tot_cur_bal"] = [sub_grade_pred.tot_cur_bal]
    sub_df["open_acc_6m"] = [sub_grade_pred.open_acc_6m]
    sub_df["open_act_il"] = [sub_grade_pred.open_act_il]
    sub_df["open_il_12m"] = [sub_grade_pred.open_il_12m]
    sub_df["open_il_24m"] = [sub_grade_pred.open_il_24m]
    sub_df["mths_since_rcnt_il"] = [sub_grade_pred.mths_since_rcnt_il]
    sub_df["total_bal_il"] = [sub_grade_pred.total_bal_il]
    sub_df["il_util"] = [sub_grade_pred.il_util]
    sub_df["open_rv_12m"] = [sub_grade_pred.open_rv_12m]
    sub_df["open_rv_24m"] = [sub_grade_pred.open_rv_24m]
    sub_df["max_bal_bc"] = [sub_grade_pred.max_bal_bc]
    sub_df["all_util"] = [sub_grade_pred.all_util]
    sub_df["total_rev_hi_lim"] = [sub_grade_pred.total_rev_hi_lim]
    sub_df["inq_fi"] = [sub_grade_pred.inq_fi]
    sub_df["total_cu_tl"] = [sub_grade_pred.total_cu_tl]
    sub_df["inq_last_12m"] = [sub_grade_pred.inq_last_12m]
    sub_df["acc_open_past_24mths"] = [sub_grade_pred.acc_open_past_24mths]
    sub_df["avg_cur_bal"] = [sub_grade_pred.avg_cur_bal]
    sub_df["bc_open_to_buy"] = [sub_grade_pred.bc_open_to_buy]
    sub_df["bc_util"] = [sub_grade_pred.bc_util]
    sub_df["mo_sin_old_il_acct"] = [sub_grade_pred.mo_sin_old_il_acct]
    sub_df["mo_sin_old_rev_tl_op"] = [sub_grade_pred.mo_sin_old_rev_tl_op]
    sub_df["mo_sin_rcnt_rev_tl_op"] = [sub_grade_pred.mo_sin_rcnt_rev_tl_op]
    sub_df["mo_sin_rcnt_tl"] = [sub_grade_pred.mo_sin_rcnt_tl]
    sub_df["mort_acc"] = [sub_grade_pred.mort_acc]
    sub_df["mths_since_recent_bc"] = [sub_grade_pred.mths_since_recent_bc]
    sub_df["mths_since_recent_bc_dlq"] = [sub_grade_pred.mths_since_recent_bc_dlq]
    sub_df["mths_since_recent_inq"] = [sub_grade_pred.mths_since_recent_inq]
    sub_df["mths_since_recent_revol_delinq"] = [
        sub_grade_pred.mths_since_recent_revol_delinq
    ]
    sub_df["num_accts_ever_120_pd"] = [sub_grade_pred.num_accts_ever_120_pd]
    sub_df["num_actv_bc_tl"] = [sub_grade_pred.num_actv_bc_tl]
    sub_df["num_actv_rev_tl"] = [sub_grade_pred.num_actv_rev_tl]
    sub_df["num_bc_sats"] = [sub_grade_pred.num_bc_sats]
    sub_df["num_bc_tl"] = [sub_grade_pred.num_bc_tl]
    sub_df["num_il_tl"] = [sub_grade_pred.num_il_tl]
    sub_df["num_op_rev_tl"] = [sub_grade_pred.num_op_rev_tl]
    sub_df["num_rev_accts"] = [sub_grade_pred.num_rev_accts]
    sub_df["num_rev_tl_bal_gt_0"] = [sub_grade_pred.num_rev_tl_bal_gt_0]
    sub_df["num_sats"] = [sub_grade_pred.num_sats]
    sub_df["num_tl_90g_dpd_24m"] = [sub_grade_pred.num_tl_90g_dpd_24m]
    sub_df["num_tl_op_past_12m"] = [sub_grade_pred.num_tl_op_past_12m]
    sub_df["pct_tl_nvr_dlq"] = [sub_grade_pred.pct_tl_nvr_dlq]
    sub_df["percent_bc_gt_75"] = [sub_grade_pred.percent_bc_gt_75]
    sub_df["pub_rec_bankruptcies"] = [sub_grade_pred.pub_rec_bankruptcies]
    sub_df["tot_hi_cred_lim"] = [sub_grade_pred.tot_hi_cred_lim]
    sub_df["total_bal_ex_mort"] = [sub_grade_pred.total_bal_ex_mort]
    sub_df["total_bc_limit"] = [sub_grade_pred.total_bc_limit]
    sub_df["total_il_high_credit_limit"] = [sub_grade_pred.total_il_high_credit_limit]
    sub_df["disbursement_method"] = [sub_grade_pred.disbursement_method]
    sub_df["Year_issue_d"] = [sub_grade_pred.Year_issue_d]
    sub_df["cos_Month_issue_d"] = [sub_grade_pred.cos_Month_issue_d]
    sub_df["sin_Month_issue_d"] = [sub_grade_pred.sin_Month_issue_d]
    sub_df["log_loan_amnt"] = [sub_grade_pred.log_loan_amnt]
    sub_df["risk_score"] = [sub_grade_pred.risk_score]

    prediction = grade_predictor.predict(sub_df)
    if prediction[0] == 0:
        prediction = "You are likely to get grade 'A1'"
    elif prediction[0] == 1:
        prediction = "You are likely to get grade 'A2'"
    elif prediction[0] == 2:
        prediction = "You are likely to get grade 'A3'"
    elif prediction[0] == 3:
        prediction = "You are likely to get grade 'A4'"
    elif prediction[0] == 4:
        prediction = "You are likely to get grade 'A5'"
    elif prediction[0] == 5:
        prediction = "You are likely to get grade 'B1'"
    elif prediction[0] == 6:
        prediction = "You are likely to get grade 'B2'"
    elif prediction[0] == 7:
        prediction = "You are likely to get grade 'B3'"
    elif prediction[0] == 8:
        prediction = "You are likely to get grade 'B4'"
    elif prediction[0] == 9:
        prediction = "You are likely to get grade 'B5'"
    elif prediction[0] == 10:
        prediction = "You are likely to get grade 'C1'"
    elif prediction[0] == 11:
        prediction = "You are likely to get grade 'C2'"
    elif prediction[0] == 12:
        prediction = "You are likely to get grade 'C3'"
    elif prediction[0] == 13:
        prediction = "You are likely to get grade 'C4'"
    elif prediction[0] == 14:
        prediction = "You are likely to get grade 'C5'"
    elif prediction[0] == 15:
        prediction = "You are likely to get grade 'D1'"
    elif prediction[0] == 16:
        prediction = "You are likely to get grade 'D2'"
    elif prediction[0] == 17:
        prediction = "You are likely to get grade 'D3'"
    elif prediction[0] == 18:
        prediction = "You are likely to get grade 'D4'"
    elif prediction[0] == 19:
        prediction = "You are likely to get grade 'D5'"
    elif prediction[0] == 20:
        prediction = "You are likely to get grade 'E1'"
    elif prediction[0] == 21:
        prediction = "You are likely to get grade 'E2'"
    elif prediction[0] == 22:
        prediction = "You are likely to get grade 'E3'"
    elif prediction[0] == 23:
        prediction = "You are likely to get grade 'E4'"
    elif prediction[0] == 24:
        prediction = "You are likely to get grade 'E5'"
    elif prediction[0] == 25:
        prediction = "You are likely to get grade 'F1'"
    elif prediction[0] == 26:
        prediction = "You are likely to get grade 'F2'"
    elif prediction[0] == 27:
        prediction = "You are likely to get grade 'F3'"
    elif prediction[0] == 28:
        prediction = "You are likely to get grade 'F4'"
    elif prediction[0] == 29:
        prediction = "You are likely to get grade 'F5'"
    elif prediction[0] == 30:
        prediction = "You are likely to get grade 'G1'"
    elif prediction[0] == 31:
        prediction = "You are likely to get grade 'G2'"
    elif prediction[0] == 32:
        prediction = "You are likely to get grade 'G3'"
    elif prediction[0] == 33:
        prediction = "You are likely to get grade 'G4'"
    else:
        prediction = "You are likely to get grade 'G5'"

    return {"prediction": prediction}


@app.post("/interest_rate_prediction")
async def create_application(int_rate_pred: InterestRatePrediction):
    rate_df = pd.DataFrame()

    if int_rate_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(
            status_code=404,
            detail='This home ownership status is not found. Please choose from "MORTGAGE", "OWN", "RENT", "ANY", "NONE", "OTHER"',
        )

    if int_rate_pred.verification_status_joint not in verification_status_joint_dict:
        raise HTTPException(
            status_code=404,
            detail='Please select a valid verification status joint from : "Not Verified", "Verified", "Source Verified"',
        )

    rate_df["home_ownership"] = [int_rate_pred.home_ownership]
    rate_df["delinq_2yrs"] = [int_rate_pred.home_ownership]
    rate_df["mths_since_last_record"] = [int_rate_pred.home_ownership]
    rate_df["open_acc"] = [int_rate_pred.home_ownership]
    rate_df["pub_rec"] = [int_rate_pred.home_ownership]
    rate_df["total_rec_late_fee"] = [int_rate_pred.home_ownership]
    rate_df["recoveries"] = [int_rate_pred.home_ownership]
    rate_df["collection_recovery_fee"] = [int_rate_pred.home_ownership]
    rate_df["mths_since_last_major_derog"] = [int_rate_pred.home_ownership]
    rate_df["verification_status_joint"] = [int_rate_pred.home_ownership]
    rate_df["tot_coll_amt"] = [int_rate_pred.home_ownership]
    rate_df["open_acc_6m"] = [int_rate_pred.home_ownership]
    rate_df["open_act_il"] = [int_rate_pred.home_ownership]
    rate_df["open_il_12m"] = [int_rate_pred.home_ownership]
    rate_df["open_il_24m"] = [int_rate_pred.home_ownership]
    rate_df["open_rv_12m"] = [int_rate_pred.home_ownership]
    rate_df["inq_fi"] = [int_rate_pred.home_ownership]
    rate_df["total_cu_tl"] = [int_rate_pred.home_ownership]
    rate_df["mort_acc"] = [int_rate_pred.home_ownership]
    rate_df["mths_since_recent_bc_dlq"] = [int_rate_pred.home_ownership]
    rate_df["mths_since_recent_revol_delinq"] = [int_rate_pred.home_ownership]
    rate_df["num_accts_ever_120_pd"] = [int_rate_pred.home_ownership]
    rate_df["num_actv_bc_tl"] = [int_rate_pred.home_ownership]
    rate_df["num_actv_rev_tl"] = [int_rate_pred.home_ownership]
    rate_df["num_bc_sats"] = [int_rate_pred.home_ownership]
    rate_df["num_op_rev_tl"] = [int_rate_pred.home_ownership]
    rate_df["num_rev_tl_bal_gt_0"] = [int_rate_pred.home_ownership]
    rate_df["num_sats"] = [int_rate_pred.home_ownership]
    rate_df["num_tl_90g_dpd_24m"] = [int_rate_pred.home_ownership]
    rate_df["pub_rec_bankruptcies"] = [int_rate_pred.home_ownership]

    prediction = interest_rate_predictor.predict(rate_df)
    rounded_prediction = round(prediction[0], 2)

    return {"prediction": rounded_prediction}
