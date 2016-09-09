# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from __future__ import division

from collections import defaultdict
import random
import json

import numpy

from nupic.engine import Network
from nupic.encoders.scalar import ScalarEncoder
from nupic.encoders.multi import MultiEncoder
from htmresearch.support.register_regions import registerAllResearchRegions


_VERBOSITY = 0

SP_PARAMS = {
  "spVerbosity": _VERBOSITY,
  "spatialImp": "cpp",
  "columnCount": 2048,
  "inputWidth": 0,
  "seed": 1936,
  "potentialPct": 0.8,
  "potentialRadius": 2048,
  "globalInhibition": 1,
  "numActiveColumnsPerInhArea": 40,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 1.0,
}

TM_PARAMS = {
  "columnCount": 2048,
  "cellsPerColumn": 8,
  "formInternalConnections": 1,
  "formInternalBasalConnections": 1,  # inconsistency between CPP and PY
  "learningMode": 1,
  "inferenceMode": 1,
  "learnOnOneCell": 0,
  "initialPermanence": 0.51,
  "connectedPermanence": 0.6,
  "permanenceIncrement": 0.1,
  "permanenceDecrement": 0.001,
  "minThreshold": 20,
  "predictedSegmentDecrement": 0.002,
  "activationThreshold": 20,
  "maxNewSynapseCount": 30,
  "monitor": 0,
  "implementation": "cpp",
}



class HTMLearner(object):
  """
  Only supports Discrete action spaces and Box observation spaces.
  """

  def __init__(self, environment, alpha=0.3,
               epsilon=0.75, eps_decay=0.99, discount=0.95):
    self.n = 2048 * 8
    self.iterations = 0
    self.environment = environment
    self.actions = range(environment.action_space.n)
    self.discount = discount
    self.weights = defaultdict(lambda: numpy.zeros(self.n))
    self.epsilon = epsilon
    self.eps_decay = eps_decay
    self.network = self._createNetwork()
    self.alpha = alpha
    self.seq = 0


  def bestAction(self, state):
    bestActions = []
    maxQValue = float("-inf")

    if random.random() < self.epsilon:
      return random.choice(self.actions)

    for action in self.actions:
      qValue = self._qValue(state, action)

      if qValue > maxQValue:
        bestActions = [action]
        maxQValue = qValue
      elif qValue == maxQValue:
        bestActions.append(action)

    return random.choice(bestActions) if len(bestActions) else None


  def update(self, state, action, nextState, reward):
    self.iterations += 1
    targetValue = reward + (self.discount * self._value(nextState))
    qValue = self._qValue(state, action)
    if sum(state) > 0:
      correction = (targetValue - qValue) / sum(state)
    else:
      correction = 0.

    for i in state.nonzero()[0]:
      self.weights[action][i] += self.alpha * correction


  def compute(self, observation, action):
    obs = dict([(str(i), observation[i]) for i in xrange(len(observation))])
    encodedObservation = self.observationEncoder.encode(obs)
    encodedAction = self.actionEncoder[action]
    self.observationSensor.addDataToQueue(
      list(encodedObservation.nonzero()[0]), 0, self.seq
    )
    self.actionSensor.addDataToQueue(list(encodedAction), 0, self.seq)
    self.network.run(1)
    return self.getState()


  def getState(self):
    return self.temporalMemoryRegion.activeState * \
           self.temporalMemoryRegion.previouslyPredictedCells


  def _qValue(self, state, action):
    qValue = 0
    for i in state.nonzero()[0]:
      qValue += self.weights[action][i]
    return qValue


  def _value(self, state):
    qValues = [self._qValue(state, action) for action in self.actions]
    return max(qValues) if len(qValues) else 0.0


  def _createNetwork(self):
    registerAllResearchRegions()
    network = Network()

    network.addRegion("observationSensor", "py.RawSensor",
                      json.dumps({"verbosity": _VERBOSITY}))
    self.observationSensor = network.regions["observationSensor"].getSelf()
    self.observationEncoder = self._createObservationEncoder()

    network.addRegion("actionSensor", "py.RawSensor",
                      json.dumps({"verbosity": _VERBOSITY}))
    self.actionSensor = network.regions["actionSensor"].getSelf()
    self.actionEncoder = dict(
      [(a, self._generatePattern()) for a in self.actions]
    )

    SP_PARAMS["inputWidth"] = self.observationEncoder.getWidth()
    network.addRegion("spatialPoolerRegion", "py.SPRegion",
                      json.dumps(SP_PARAMS))
    network.addRegion("temporalMemoryRegion", "py.ExtendedTMRegion",
                      json.dumps(TM_PARAMS))

    network.link("observationSensor", "spatialPoolerRegion", "UniformLink", "",
                 srcOutput="dataOut", destInput="bottomUpIn")
    network.link("observationSensor", "spatialPoolerRegion", "UniformLink", "",
                 srcOutput="resetOut", destInput="resetIn")
    network.link("spatialPoolerRegion", "temporalMemoryRegion", "UniformLink",
                 "")
    network.link("actionSensor", "temporalMemoryRegion", "UniformLink", "",
                 srcOutput="dataOut", destInput="externalInput")

    self.spatialPoolerRegion = network.regions[
      "spatialPoolerRegion"
    ].getSelf()
    self.temporalMemoryRegion = network.regions[
      "temporalMemoryRegion"
    ].getSelf()

    self.spatialPoolerRegion.setParameter("learningMode", 1, 1)
    self.temporalMemoryRegion.setParameter("learningMode", 1, 1)

    return network


  def _createObservationEncoder(self):
    """
    Only works with Box observation spaces.

    Hacky modifications for Cartpole.
    """
    multiEncoder = MultiEncoder()
    for i in xrange(self.environment.observation_space.shape[0]):
      high = self.environment.observation_space.high[i]
      low = self.environment.observation_space.low[i]
      # hacky
      if i == 1 or 3:
        high = 2.2
        low = -2.2
      if i == 2:
        high /= 2.
        low /= 2.

      encoder = ScalarEncoder(41, low, high, n=512, name=str(i),
                              clipInput=True)
      multiEncoder.addEncoder(str(i), encoder)

    return multiEncoder


  def _generatePattern(self):
    cellsIndices = range(2048)
    random.shuffle(cellsIndices)
    return set(cellsIndices[:40])


  def _reset(self):
    self.iterations = 0
    self.observationSensor.addResetToQueue(self.seq)
    self.actionSensor.addResetToQueue(self.seq)
    self.network.run(1)
    self.temporalMemoryRegion.reset()
    self.seq += 1
