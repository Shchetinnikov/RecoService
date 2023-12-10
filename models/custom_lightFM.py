import pandas as pd
import numpy as np

from rectools.dataset import Dataset
from rectools.models import PopularModel, ImplicitALSWrapperModel, LightFMWrapperModel
from rectools.tools import UserToItemAnnRecommender

from lightfm import LightFM


# Имплементация LightFM, PopularModel, ANN
class CustomLightFMWithANN():
    def __init__(self, params: dict, n_epochs: int = 10, n_threads: int = 4):
        self.lightFM = LightFMWrapperModel(
            model=LightFM(**params),
            epochs=n_epochs,
            num_threads=n_threads
        )
        self.pop_model = PopularModel()
        self.ann_model = None
        self.is_fitted = False

    def get_mappings(self, train):
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def fit(self, dataset: Dataset):
        train = dataset.interactions.df
        self.get_mappings(train)

        # обучаем PopularModel, LightFM, ANN
        self.lightFM.fit(dataset)
        self.pop_model.fit(dataset)

        user_vectors, item_vectors = self.lightFM.get_vectors(dataset)

        self.ann_model = UserToItemAnnRecommender(
            user_vectors=user_vectors,
            item_vectors=item_vectors,
            user_id_map=dataset.user_id_map,
            item_id_map=dataset.item_id_map,
        )

        self.ann_model.fit()

        # список популярного
        popular_list = [self.users_inv_mapping[item] for item in
                        self.pop_model.popularity_list[0]]
        popular_scores = self.pop_model.popularity_list[1]
        self.popular = pd.DataFrame([popular_list, popular_scores]).transpose() \
            .rename(columns={0: "item_id", 1: "score"}) \
            .reset_index(drop=True)

        self.is_fitted = True

    def recommend(self, users: np.array, dataset: Dataset, k: int = 10,
                  filter_viewed: bool = False):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        # выделяем "горячих" и "холодных" юзеров
        user_ids = dataset.user_id_map.external_ids
        hot_users_mask = np.isin(users, user_ids)
        cold_users = users[~hot_users_mask]
        hot_users = users[hot_users_mask]

        # рекомендуем "холодным" юзерам популярное
        cold_users = pd.DataFrame(cold_users, columns=['user_id'])
        pop_recs = pd.merge(cold_users, self.popular.iloc[:k], how='cross')
        pop_recs['rank'] = pop_recs.groupby('user_id').cumcount() + 1
        pop_recs.drop(columns=['score'], inplace=True)

        # ann рекомендации
        ann_recs = self.ann_model.get_item_list_for_user_batch(
            user_ids=hot_users, top_n=k)
        ann_recs = pd.DataFrame({'user_id': hot_users,
                                 'item_id': ann_recs}).explode('item_id')
        ann_recs['rank'] = ann_recs.groupby('user_id').cumcount() + 1

        resc = pd.concat([ann_recs, pop_recs])
        resc = resc.astype({"user_id": int, "item_id": int, "rank": int})
        return resc

    def predict_single(self, user_id: int, N_recs: int = 10):
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")
        try:
            recs = self.ann_model.get_item_list_for_user(user_id=user_id,
                                                         top_n=N_recs)
        except:
            recs = self.popular.iloc[:N_recs]
        return recs
