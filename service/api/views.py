from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

<<<<<<< HEAD
from service.api.exceptions import UserNotFoundError, ModelNotFoundError
=======
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
>>>>>>> b87211a7988c72a788e2b13c99801abd40112b8b
from service.log import app_logger


class ErrorMessage(BaseModel):
    msg: str


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        200: {
            "description": "Successful request",
            "model": RecoResponse
        },
        404: {
            "description": "Request error",
            "model": ErrorMessage
        }
    }
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Write your code here
    if model_name == "range_model":
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
