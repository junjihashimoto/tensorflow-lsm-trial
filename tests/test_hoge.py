from hoge import Hoge
import pytest


def test_hoge():
  obj = Hoge.Piyo()
  assert 1 == obj.one()
