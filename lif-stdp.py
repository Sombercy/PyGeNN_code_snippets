#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:21:55 2020

@author: xana
"""

import numpy as np
from os import path
from collections import Counter
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class,
                               create_custom_weight_update_class,
                               create_custom_init_var_snippet_class,
                               create_dpf_class,
                               init_var,
                               GeNNModel)
from pygenn.genn_wrapper import NO_DELAY

def record_current_spikes(pop, spikes, dt):
    current_spikes = pop.current_spikes
    current_spike_times = np.ones(current_spikes.shape) * dt

    if spikes is None:
        return (np.copy(current_spikes), current_spike_times)
    else:
        return (np.hstack((spikes[0], current_spikes)),
                np.hstack((spikes[1], current_spike_times)))


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DT = 1.0

INPUT_CURRENT_SCALE = 1.0 / 100.0

G_MAX = 0.01
DURATION_MS = 300.0
NUM_MBON = 10

A_PLUS = 0.01
A_MINUS = 1.05 * A_PLUS
# ----------------------------------------------------------------------------
# Custom GeNN models
# ----------------------------------------------------------------------------
# Very simple integrate-and-fire neuron model
if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vthr"],
    var_name_types=[("V", "scalar"), ("SpikeCount", "unsigned int")],
    sim_code="$(V) += $(Isyn) * DT;",
    reset_code="""
    $(V) = 0.0;
    $(SpikeCount)++;
    """,
    threshold_condition_code="$(V) >= $(Vthr)")

# Current source model which injects current with a magnitude specified by a state variable
cs_model = create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# Model for graded synapses with exponential activation
graded_synapse_model = create_custom_weight_update_class(
    "graded_synapse_model",
    param_names=["Vmid", "Vslope", "Vthresh"],
    var_name_types=[("g", "scalar")],
    event_code="$(addToInSyn, DT * $(g) * fmax(0.0, 1.0 / (1.0 + exp(($(Vmid) - $(V_pre)) / $(Vslope)))));",
    event_threshold_condition_code="$(V_pre) > $(Vthresh)")

lateral_inhibition = create_custom_init_var_snippet_class(
    "lateral_inhibition",
    param_names=["g"],
    var_init_code="$(value)=($(id_pre)==$(id_post)) ? 0.0 : $(g);")

# STDP synapse with multiplicative weight dependence
stdp_multiplicative = create_custom_weight_update_class(
    "STDPMultiplicative",
    param_names=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],

    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauMinus));
            $(g) -= ($(g) - $(wMin)) * $(aMinus) * $(postTrace) * timing;
        }
        """,

    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauPlus));
            $(g) += ($(wMax) - $(g)) * $(aPlus) * $(preTrace) * timing;
        }
        """,

    pre_spike_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        """,

    post_spike_code=
        """
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        """,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)

# STDP synapse with additive weight dependence
symmetric_stdp = create_custom_weight_update_class(
    "symmetric_stdp",
    param_names=["tau", "rho", "eta", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        const scalar timing = exp(-dt / $(tau));
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)
# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "song")
model.dT = DT

if_params = {"Vthr": 5.0}
if_init = {"V": 0.0, "SpikeCount":0}

lif_params = {"C": 0.17, "TauM": 10.0, "Vrest": -74.0, "Vreset": -60.0,
              "Vthresh": -54.0, "Ioffset": 0.0, "TauRefrac": 1.0}

lif_init = {"V": -60.0, "RefracTime": 0.0}

post_syn_params = {"tau": 5.0}

stdp_init = {"g": init_var("Uniform", {"min": 0.0, "max": G_MAX})}
stdp_params = {"tauPlus": 20.0, "tauMinus": 20.0, "aPlus": A_PLUS, "aMinus": A_MINUS, "wMin": 0.0, "wMax": G_MAX}
stdp_pre_init = {"preTrace": 0.0}
stdp_post_init = {"postTrace": 0.0}

# Load weights
weights = []
while True:
    filename = "weights_%u_%u.npy" % (len(weights), len(weights) + 1)
    if path.exists(filename):
        weights.append(np.load(filename))
    else:
        break

#weights[0] = np.random.rand(784, 128)
#weights[1] = np.random.rand(128,10)
weights[1] = np.zeros([128, 10])

# Create first neuron layer
neuron_layers = [model.add_neuron_population("neuron0", weights[0].shape[0],
                                             if_model, if_params, if_init)]
# Create subsequent neuron layer
for i, w in enumerate(weights):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i + 1),
                                                     w.shape[1], "LIF",
                                                     lif_params, lif_init))
    
# Create synaptic connections between layers
for i, (pre, post, w) in enumerate(zip(neuron_layers[:-1], neuron_layers[1:], weights)):
    if (i == 0):
        model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            "StaticPulse", {}, {"g": w.flatten()}, {}, {},
            "DeltaCurr", {}, {})
    else:
        model.add_synapse_population(
            "synapse%u" % i, "DENSE_INDIVIDUALG", NO_DELAY,
            pre, post,
            "NGRADSYNAPSE", stdp_params, stdp_init, stdp_pre_init, stdp_post_init,
            "ExpCurr", post_syn_params, {})
    
    
    
# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0" , {}, {"magnitude": 0.0})


# Build and load our model
model.build()
model.load()

# ----------------------------------------------------------------------------
# Simulate
# ----------------------------------------------------------------------------
# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

# Check dimensions match network
assert testing_images.shape[1] == weights[0].shape[0]
assert np.max(testing_labels) == (weights[1].shape[1] - 1)

# Set current input by scaling first image
current_input.vars["magnitude"].view[:] = testing_images[0] * INPUT_CURRENT_SCALE

# Upload
model.push_var_to_device("current_input", "magnitude")

spikes = None

# Simulate
layer_spikes = [(np.empty(0), np.empty(0)) for _ in enumerate(neuron_layers)]
while model.timestep < DURATION_MS:
    # Advance simulation
    model.step_time()

    # Loop through neuron layers
    for i, l in enumerate(neuron_layers):
        # Download spikes
        model.pull_current_spikes_from_device(l.name)

        # Add to data structure
        spikes = record_current_spikes(l, spikes, model.t)

        spike_times = np.ones_like(l.current_spikes) * model.t
        layer_spikes[i] = (np.hstack((layer_spikes[i][0], l.current_spikes)), np.hstack((layer_spikes[i][1], spike_times)))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Create a plot with axes for each
fig, axes = plt.subplots(len(neuron_layers), sharex=True)


# Loop through axes and their corresponding neuron populations
for a, s, l in zip(axes, layer_spikes, neuron_layers):
    # Plot spikes
    a.scatter(s[1], s[0], s=1)

    # Set title, axis labels
    a.set_title(l.name)
    a.set_ylabel("Spike number")
    a.set_xlim((0, DURATION_MS * DT))
    a.set_ylim((0, l.size))


# Add an x-axis label and translucent line showing the correct label
axes[-1].set_xlabel("Time [ms]")
axes[-1].hlines(testing_labels[0], xmin=0, xmax=DURATION_MS, linestyle="--", color="gray", alpha=0.2)

# Show plot
plt.show()

# ----------------------------------------------------------------------------
# Simulate
# ----------------------------------------------------------------------------
# Load testing data
testing_images = np.load("testing_images.npy")
testing_labels = np.load("testing_labels.npy")

# Check dimensions match network
assert testing_images.shape[1] == weights[0].shape[0]
assert np.max(testing_labels) == (weights[1].shape[1] - 1)

# Get views to efficiently access state variables
current_input_magnitude = current_input.vars["magnitude"].view
output_spike_count = None
layer_voltages = [l.vars["V"].view for l in neuron_layers]
label_spikes = np.zeros((NUM_MBON, NUM_MBON), dtype=int)

# Simulate
num_correct = 0
while model.timestep < (DURATION_MS * testing_images.shape[0]):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % DURATION_MS
    example = int(model.timestep // DURATION_MS)

    # If this is the first timestep of presenting the example
    if timestep_in_example == 0:
        current_input_magnitude[:] = testing_images[example] * INPUT_CURRENT_SCALE
        model.push_var_to_device("current_input", "magnitude")

        # Loop through all layers and their corresponding voltage views
        for l, v in zip(neuron_layers, layer_voltages):
            # Manually 'reset' voltage
            v[:] = 0.0
            
            # Upload
            model.push_var_to_device(l.name, "V")
    
    # Advance simulation
    model.step_time()
    
    # Download spikes
    model.pull_current_spikes_from_device("neuron2")
    label_spikes[testing_labels[example]] += np.bincount(neuron_layers[2].current_spikes, minlength=NUM_MBON)
    
    # If this is the LAST timestep of presenting the example
    if timestep_in_example == (DURATION_MS - 1):
        # Find which neuron spiked the most to get prediction
        
        predicted_label = np.argmax(label_spikes[testing_labels[example]])
        true_label = testing_labels[example]

        print("\tExample=%u, true label=%u, predicted label=%u" % (example,
                                                                   true_label,
                                                                   predicted_label))
        
        if predicted_label == true_label:
            num_correct += 1
    
print("Accuracy %f%%" % ((num_correct / float(testing_images.shape[0])) * 100.0))


