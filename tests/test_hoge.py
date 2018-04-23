from hoge import Hoge
import pytest

@pytest.mark.django_db(transaction=True)
def test_hoge():
  obj = Hoge.Piyo()
  assert 1 == obj.one()
