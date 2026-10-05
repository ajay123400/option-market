"""Builds a full fake trading day (fake clock + fake Fyers world, a spot drift and one HTTP 429) into a data directory, for independent_check.py. Usage (repo root): python collector/validation/make_fake_day.py DIR"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.collector_fakes import *
from collector import config as C
from collector.recorder import Recorder
from collector.sources import TokenProvider
out = sys.argv[1]
shutil.rmtree(out, ignore_errors=True)
clk = FakeClock(start(9, 10)); w = World(clk); clk.hooks.append(w.emit)
state = {"n": 0}
def move():                                                   # make the day interesting: the spot drifts and one 429 happens
    if clk.now() >= start(12, 0): w.spot = 22640.0
    if clk.now() >= start(14, 0): w.spot = 22380.0
clk.hooks.append(move)
w.rate_limit_on = {40}
rec = Recorder(C.Config(), out, TokenProvider(FakeAuth()), clk, clk.stop, rest_session=w.session(), ws_factory=w.factory(), ws_spawn=lambda f: f())
print("rc", rec.run())
