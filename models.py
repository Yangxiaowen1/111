from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone
from uuid import uuid4


class UserProfile(models.Model):
    """本地用户基础信息，供行为表引用。"""

    GENDER_CHOICES = (
        ("male", "男"),
        ("female", "女"),
        ("unknown", "未知"),
    )

    external_id = models.UUIDField(
        default=uuid4,
        unique=True,
        editable=False,
        help_text="对外暴露的用户唯一 ID",
    )
    username = models.CharField(max_length=50, unique=True)
    nickname = models.CharField(max_length=50, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default="unknown")
    city = models.CharField(max_length=50, blank=True)
    avatar = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "user_profile"
        verbose_name = "用户信息"
        verbose_name_plural = "用户信息"

    def __str__(self) -> str:
        return self.username


class BaseBehavior(models.Model):
    """用户行为抽象基类，包含公共字段。"""

    user = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="%(class)s_records",
    )
    poi_id = models.BigIntegerField(help_text="关联的旅游景点 ID（Hive 数据源）")
    scene = models.CharField(max_length=32, blank=True, help_text="触发场景，例如首页/搜索结果")
    device = models.CharField(max_length=32, blank=True, help_text="客户端设备信息")
    occurred_at = models.DateTimeField(default=timezone.now)

    class Meta:
        abstract = True


class UserClickBehavior(BaseBehavior):
    """用户在列表页/推荐位的点击行为。"""

    exposure_id = models.CharField(
        max_length=64,
        blank=True,
        help_text="可选的曝光批次 ID，方便召回评估",
    )

    class Meta:
        db_table = "user_click_behavior"
        verbose_name = "用户点击行为"
        verbose_name_plural = "用户点击行为"


class UserDetailViewBehavior(BaseBehavior):
    """用户查看景点详情页的行为记录。"""

    stay_seconds = models.PositiveIntegerField(default=0, help_text="本次停留时长，单位秒")
    from_click = models.ForeignKey(
        UserClickBehavior,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="detail_views",
        help_text="如果详情来自某次点击，可关联点击记录",
    )

    class Meta:
        db_table = "user_detail_view_behavior"
        verbose_name = "用户查看详情行为"
        verbose_name_plural = "用户查看详情行为"


class UserFavoriteBehavior(BaseBehavior):
    """用户收藏/取消收藏景点的行为。"""

    is_favorite = models.BooleanField(default=True, help_text="True 表示收藏，False 表示取消")
    note = models.CharField(max_length=200, blank=True)

    class Meta:
        db_table = "user_favorite_behavior"
        verbose_name = "用户收藏行为"
        verbose_name_plural = "用户收藏行为"


class UserCommentBehavior(BaseBehavior):
    """用户对景点的评论及评分。"""

    rating = models.PositiveSmallIntegerField(
        default=0,
        validators=[MinValueValidator(0), MaxValueValidator(5)],
        help_text="用户评分 0-5",
    )
    content = models.TextField(blank=True)
    image_urls = models.TextField(
        blank=True,
        help_text="JSON 字符串形式存放评论配图 URL 列表，兼容 SQLite",
    )

    class Meta:
        db_table = "user_comment_behavior"
        verbose_name = "用户评论行为"
        verbose_name_plural = "用户评论行为"
