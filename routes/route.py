from fastapi import APIRouter, HTTPException, Query
from Models.Charts import Chart
from Tools.PTForcasting import PTForcasting
from Tools.THDAForecasting import THDAForecasting
from schema.schemas import list_serial
from Tools.STForcasting import STForcasting

router = APIRouter()

# Handlers are sync so FastAPI runs the blocking pandas/XGBoost work in its threadpool
@router.get("/getPTPredictions")
def get_pt_predictions(months: int = Query(1, ge=1, le=4)):
    forecasting = PTForcasting()
    try:
        forecasting.load_and_process_data(id="620d2405d2a8e7e597a36c41", field="Pt")
    except Exception as e:
        # Connection refused, bad token, etc. -- report it as the upstream being unavailable
        raise HTTPException(status_code=503, detail=f"InfluxDB unavailable: {type(e).__name__}")
    forecasting.split_data()
    forecasting.feature_engineering()
    forecasting.load_model()
    future_df, threshold = forecasting.forecast(months=months)

    return {"PTPredictions": future_df, "threshold": threshold}

@router.get("/getTHDAPredictions")
def get_thda_predictions(months: int = Query(1, ge=1, le=4)):
    forecasting = THDAForecasting('routes/thda.csv')
    forecasting.load_and_process_data()
    forecasting.split_data()
    forecasting.feature_engineering()
    forecasting.load_model()
    future_df, threshold = forecasting.forecast(months=months)
    return {"THDAPredictions": future_df, "THDAthreshold": threshold}

@router.get("/getSTPredictions")
def get_st_predictions(months: int = Query(1, ge=1, le=4)):
    forecasting = STForcasting('routes/st.csv')
    forecasting.load_and_process_data()
    forecasting.split_data()
    forecasting.feature_engineering()
    forecasting.load_model()
    future_df, threshold = forecasting.forecast(months=months)
    return {"STPredictions": future_df, "STthreshold": threshold}
