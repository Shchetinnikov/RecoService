import pandas as pd
import numpy as np

from rectools.dataset import Dataset
from rectools.models import PopularModel
from implicit.nearest_neighbours import ItemItemRecommender
from models.userknn import UserKnn


# Имплементация UserKnn и PopularModel
class CustomUserKnn():
  def __init__(self, model: ItemItemRecommender, N_users: int = 50):
    self.N_users = N_users
    self.user_knn = UserKnn(model, N_users)
    self.pop_model = PopularModel()
    self.is_fitted = False

  def get_mappings(self, train):
    self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
    self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

    self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
    self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

  def fit(self, dataset: Dataset):
    train = dataset.interactions.df
    self.get_mappings(train)

    # обучаем PopularModel и UserKnn
    self.user_knn.fit(train)
    self.pop_model.fit(dataset)

    # список популярного
    popular_list = [self.users_inv_mapping[item] for item in self.pop_model.popularity_list[0]]
    popular_scores = self.pop_model.popularity_list[1]
    self.popular = pd.DataFrame([popular_list, popular_scores]).transpose()\
                     .rename(columns={0: "item_id", 1: "score"})\
                     .reset_index(drop=True)

    self.is_fitted = True

  def recommend(self, users: pd.DataFrame, dataset: Dataset, N_recs: int = 10):
    if not self.is_fitted:
            raise ValueError("Please call fit before predict")

    # выделяем "горячих" и "холодных" юзеров
    interactions = dataset.interactions.df
    hot_users_mask = users['user_id'].isin(interactions['user_id'].unique())
    cold_users = users[~hot_users_mask][Columns.User]
    hot_users = users[hot_users_mask]

    # рекомендуем "холодным" юзерам популярное
    cold_users = pd.DataFrame(cold_users)
    pop_recs = pd.merge(cold_users, self.popular.iloc[:N_recs], how='cross')
    pop_recs['rank'] = pop_recs.groupby('user_id').cumcount() + 1

    # user_knn рекомендации 
    knn_recs = self.user_knn.predict(hot_users, N_recs)

    # Попытка задать ровно N рекомендаций (долго)
    # for hot_user_id in hot_users['user_id'].unique():
    #   user_recs = knn_recs[knn_recs['user_id']==hot_user_id]
    #   if len(user_recs) < N_recs:
    #     new_recs = self.popular[:N_recs - len(user_recs)]
    #     new_recs['user_id'] = hot_user_id
    #     knn_recs = pd.concat([knn_recs, new_recs])

    knn_recs = knn_recs.sort_values(['user_id', 'score'], ascending=False)
    knn_recs['rank'] = knn_recs.groupby('user_id').cumcount() + 1
    
    resc = pd.concat([knn_recs, pop_recs])
    resc = resc.astype({"user_id": int, "item_id": int, "score": float, "rank": int})
    return resc

  def predict_single(self, user_id: int, N_recs: int = 10):
    if not self.is_fitted:
        raise ValueError("Please call fit before predict")

    user_id_mapped = self.users_mapping.get(user_id)
    users, sim = self.user_knn.model.similar_items(user_id_mapped, N=self.N_users)

    recs = []
    watched = self.user_knn.watched
    for sim_user, similarity in zip(users, sim):
        sim_user_id = self.users_inv_mapping.get(sim_user)
        if sim_user_id is not None and sim_user_id != user_id:
          watched_items = watched.loc[watched['sim_user_id'] == sim_user_id, 'item_id'].values
          watched_items_flat = np.concatenate(watched_items)

          watched_items_flat = np.unique(watched_items_flat[watched_items_flat != None])
          recs.extend(filter(None, map(self.items_mapping.get, watched_items_flat)))

    recs = list(set(recs))[:N_recs]
    return recs