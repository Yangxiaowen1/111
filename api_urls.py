from django.urls import path

from .views import ClickView, CommentView, DetailViewLogView, FavoriteView

app_name = "user_api"

urlpatterns = [
    path("click/", ClickView.as_view(), name="click"),
    path("favorites/", FavoriteView.as_view(), name="favorites"),
    path("detail-view/", DetailViewLogView.as_view(), name="detail-view"),
    path("comments/", CommentView.as_view(), name="comments"),
]

