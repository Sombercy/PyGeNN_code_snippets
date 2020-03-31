import numpy as np
import matplotlib.pyplot as plt
from os import path
from struct import unpack

from gzip import decompress
from urllib import request
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_current_source_class,
                               create_custom_weight_update_class,
                               create_dpf_class,
                               create_custom_init_var_snippet_class,
                               create_custom_postsynaptic_class,
                               init_var,
                               GeNNModel,
                               init_connectivity)
from pygenn.genn_wrapper import NO_DELAY

def record_current_spikes(pop, spikes, dt):
    current_spikes = pop.current_spikes
    current_spike_times = np.ones(current_spikes.shape) * dt

    if spikes is None:
        return (np.copy(current_spikes), current_spike_times)
    else:
        return (np.hstack((spikes[0], current_spikes)),
                np.hstack((spikes[1], current_spike_times)))
    

import sys

def get_image_data(url, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            image_data = decompress(response.read())

            # Unpack header from first 16 bytes of buffer
            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual images
            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            # Write to disk
            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(url, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            label_data = decompress(response.read())

            # Unpack header from first 8 bytes of buffer
            magic, num_items = unpack('>II', label_data[:8])
            assert magic == correct_magic

            # Convert remainder of buffer to numpy bytes
            label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert label_data_np.shape == (num_items,)

            # Write to disk
            np.save(filename, label_data_np)

            return label_data_np

def get_training_data():
    images = get_image_data("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "training_images.npy", 2051)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data():
    images = get_image_data("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "testing_images.npy", 2051)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

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
DELAY = 100 # 50 ms by 0.5 ms timesteps
TIMESTEP = 0.5
PRESENT_TIMESTEPS = 500
INPUT_CURRENT_SCALE = 1.0 / 100.0
G_MAX = 0.25
G_MIN = 0
INIT_WEIGHTS = {"min": G_MAX/2, "max": G_MAX}
if_params = {"Vthr": 5.0}
rm_params = {"Vspike":60, "alpha":3, "y":-2.468, "beta":0.0165}
ex_params = {"Vsyn": 0, "tau_syn": 10, "g": G_MAX}
inh_params = {"Vsyn": -92, "tau_syn": 10, "g": G_MAX}
PROBABILITY_CONNECTION = 0.75
fixed_prob = {"prob": PROBABILITY_CONNECTION}

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

lateral_inhibition = create_custom_init_var_snippet_class(
    "lateral_inhibition",
    param_names=["g"],
    var_init_code="$(value)=($(id_pre)==$(id_post)) ? 0.0 : $(g);")

# Model of conductance-based synapse Isyn = gS(V-Vsyn)
cb_synapse = create_custom_postsynaptic_class(
    "cb_synapse",
    param_names = ["Vsyn", "tau_syn", "g"],
    var_name_types = [("V", "scalar")],
    decay_code = """
    if ($(t) == $(sT_pre)) {$(inSyn)+=1;}
    else {$(inSyn) *= 1 - $(DT)/$(tau_syn);}
    """,
    apply_input_code = '$(Isyn) = g*$(inSyn)*($(V)-$(Vsyn))',
    )
                      
# STDP weight update rule
stdp = create_custom_weight_update_class(
    "stdp",
    param_names=["gMax", "gMin"],
    var_name_types=[("g", "scalar")],
    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(sT_post) - $(sT_pre);
        const scalar newG;
        if ((dt>20) && (dt<=200)) {
                newG = $(g) - 0.0125*$(inSyn);}
        else { 
            if ((dt > 2) && (dt <= 20)) {
                    newG = $(g) - 0.0117 * $(inSyn)*dt + 0.223 * $(inSyn);}
            else {if ((dt > -200) && (dt <= 2)) {
                    newG = $(g) - 0.0025 * $(inSyn);}
                else {newG = 0;}}
            }
        $(g) = fmin($(gMax), fmax($(gMin), newG));
        """,
    learn_post_code=
         """
        const scalar dt = $(sT_post) - $(sT_pre);
        const scalar newG;
        if ((dt>20) && (dt<=200)) {
                newG = $(g) - 0.0125*$(inSyn);}
        else { 
            if ((dt > 2) && (dt <= 20)) {
                    newG = $(g) - 0.0117 * $(inSyn)*dt + 0.223 * $(inSyn);}
            else {if ((dt > -200) && (dt <= 2)) {
                    newG = $(g) - 0.0025 * $(inSyn);}
                else {newG = 0;}}
            }
        $(g) = fmin($(gMax), fmax($(gMin), newG));
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)
# ----------------------------------------------------------------------------
# Build model
# ----------------------------------------------------------------------------
# Create GeNN model
model = GeNNModel("float", "simp_mnist")
model.dT = TIMESTEP

# Load weights
weights = []
while True:
    filename = "weights_%u_%u.npy" % (len(weights), len(weights) + 1)
    if path.exists(filename):
        weights.append(np.load(filename))
    else:
        break

# weights[1] = np.random.uniform(G_MAX/2, G_MAX, (128,10))

# Initial values to initialise all neurons to
if_init = {"V": 0.0, "SpikeCount":0}
rm_init = {"V": 50.0, "preV": 50.0}

# Create first neuron layer
neuron_layers = [model.add_neuron_population("neuron0", weights[0].shape[0],
                                             if_model, if_params, if_init)]

# Create current source to deliver input to first layers of neurons
current_input = model.add_current_source("current_input", cs_model,
                                         "neuron0" , {}, {"magnitude": 0.0})

# Create subsequent neuron layer
for i, w in enumerate(weights):
    neuron_layers.append(model.add_neuron_population("neuron%u" % (i + 1),
                                                     w.shape[1], "RulkovMap",
                                                     rm_params, rm_init))

# Create synaptic connections between layers
model.add_synapse_population(
    "synapse%u" % 0, "DENSE_INDIVIDUALG", NO_DELAY,
    neuron_layers[0], neuron_layers[1],
    "StaticPulse", {}, {"g": weights[0].flatten()}, {}, {},
    "DeltaCurr", {}, {})
model.add_synapse_population(
    "synapse%u" % 1, "DENSE_INDIVIDUALG", DELAY,
    neuron_layers[1], neuron_layers[2],
    stdp, {G_MAX, G_MIN}, {"g": init_var("Uniform", INIT_WEIGHTS)}, {}, {},
    cb_synapse, ex_params,  {"V": 50.0}, init_connectivity("FixedProbabilityNoAutapse", fixed_prob))
model.add_synapse_population(
    "synapse%u" % 2, "DENSE_INDIVIDUALG", DELAY,
    neuron_layers[2], neuron_layers[2],
    "StaticPulse", {}, {"g": init_var(lateral_inhibition, {"g": 0.025})}, {}, {},
    "DeltaCurr", {}, init_connectivity("FixedProbabilityNoAutapse", fixed_prob))



# Build and load our model
model.build()
model.load()
