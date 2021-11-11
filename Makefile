################################################################################
# GLOBALS
################################################################################

include gmsl

SRC_DIR = code
DATASET_DIR = datasets
HYPERPARAMCONFIG_DIR = config_files
MODELS_PARENT_DIR = trained_models/agents
DECODERS_PARENT_DIR = trained_models/decoders
PRED_SEQ_DIR = results/prediction_sequence
LR_DIR = results/learning_rate
COUPLING_DIR = results/higher_level_inference
PERF_DIR = results/performance
DECODING_DIR = results/readout
DYNAMICS_DIR = results/dynamics
PERTURBATION_DIR = results/perturbation_experiment
COMPLEXITY_DIR = results/complexity
TRAININGPLOT_DIR = results/training
HYPERPARAM_DIR = results/hyperparameter_optimization
TRAININGDYNAMICS_DIR = results/training_dynamics
RAY_DIR = trained_models/raw_training_outputs
COMPLEXITY_RAY_DIR = $(RAY_DIR)/complexity

ARCHNAME_GRU = gru
ARCHNAME_ELMAN = elman
ARCHNAME_GRUDIAG = grudiag
ARCHNAME_GRURESERVOIR = grureservoir
ARCHNAME_LAST_ELMAN = last_elman
ARCHNAME_LAST_GRURESERVOIR = last_grureservoir
ARCHNAME_LSTM = lstm

TASKNAME_BERNOULLI = B
TASKNAME_MARKOV_INDEPENDENT = MI
TASKNAME_MARKOV_COUPLED = MC
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval TASKNAMELC_$(task) = $(call lc,$(TASKNAME_$(task)))))

NETWORK_GROUP_ID_BERNOULLI = gru11_b_2020-12-11
NETWORK_GROUP_ID_MARKOV_INDEPENDENT = gru11_mi_2020-12-11
NETWORK_GROUP_ID_MARKOV_COUPLED = gru11_mc_2020-12-11
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval GRU_GROUP_ID_$(task) = $(NETWORK_GROUP_ID_$(task))))

DELTARULE_ITEM_GROUP_ID_BERNOULLI = dritem_b_20_20-09-03
DELTARULE_ITEM_GROUP_ID_MARKOV_INDEPENDENT = dritem_mi_20_20-09-03
DELTARULE_ITEM_GROUP_ID_MARKOV_COUPLED = dritem_mc_20_20-09-03

LEAKY_ITEM_GROUP_ID_BERNOULLI = leakyitem_b_20_20-09-03
LEAKY_ITEM_GROUP_ID_MARKOV_INDEPENDENT = leakyitem_mi_20_20-09-03
LEAKY_ITEM_GROUP_ID_MARKOV_COUPLED = leakyitem_mc_20_20-09-03

DELTARULE_TRANSITION_GROUP_ID_BERNOULLI = drtransition_b_20_20-09-03
DELTARULE_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT = drtransition_mi_20_20-09-03
DELTARULE_TRANSITION_GROUP_ID_MARKOV_COUPLED = drtransition_mc_20_20-09-03

LEAKY_TRANSITION_GROUP_ID_BERNOULLI = leakytransition_b_20_20-09-03
LEAKY_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT = leakytransition_mi_20_20-09-03
LEAKY_TRANSITION_GROUP_ID_MARKOV_COUPLED = leakytransition_mc_20_20-09-03

HEURISTIC_GROUP_IDS_BERNOULLI = $(DELTARULE_ITEM_GROUP_ID_BERNOULLI) \
	$(LEAKY_ITEM_GROUP_ID_BERNOULLI) \
	$(DELTARULE_TRANSITION_GROUP_ID_BERNOULLI) \
	$(LEAKY_TRANSITION_GROUP_ID_BERNOULLI)

HEURISTIC_GROUP_IDS_MARKOV_INDEPENDENT = $(DELTARULE_ITEM_GROUP_ID_MARKOV_INDEPENDENT) \
	$(LEAKY_ITEM_GROUP_ID_MARKOV_INDEPENDENT) \
	$(DELTARULE_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT) \
	$(LEAKY_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT)

HEURISTIC_GROUP_IDS_MARKOV_COUPLED = $(DELTARULE_ITEM_GROUP_ID_MARKOV_COUPLED) \
	$(LEAKY_ITEM_GROUP_ID_MARKOV_COUPLED) \
	$(DELTARULE_TRANSITION_GROUP_ID_MARKOV_COUPLED) \
	$(LEAKY_TRANSITION_GROUP_ID_MARKOV_COUPLED)

TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval MAIN_GROUP_IDS_$(task) = $(HEURISTIC_GROUP_IDS_$(task)) $(NETWORK_GROUP_ID_$(task))))

NETWORK_GROUP_IDS_PER_VOLATILITY = \
	gru11_b1by300_2020-12-28 \
    gru11_b1by150_2020-12-28 \
    gru11_b_2020-12-11 \
    gru11_b2by75_2020-12-28

NETWORK_GROUP_ID_BERNOULLI_THROUGH_TRAINING = gru11_b_through_training_2021-01-03

ELMAN_GROUP_ID_BERNOULLI = elman11_b_2020-12-11
ELMAN_GROUP_ID_MARKOV_INDEPENDENT = elman11_mi_2020-12-11
ELMAN_GROUP_ID_MARKOV_COUPLED = elman11_mc_2020-12-11

GRUDIAG_GROUP_ID_BERNOULLI = grudiag11_b_2020-12-11
GRUDIAG_GROUP_ID_MARKOV_INDEPENDENT = grudiag11_mi_2020-12-11
GRUDIAG_GROUP_ID_MARKOV_COUPLED = grudiag11_mc_2020-12-11

GRURESERVOIR_GROUP_ID_BERNOULLI = grureservoir11_b_2020-12-11
GRURESERVOIR_GROUP_ID_MARKOV_INDEPENDENT = grureservoir11_mi_2020-12-11
GRURESERVOIR_GROUP_ID_MARKOV_COUPLED = grureservoir11_mc_2020-12-11

DELTARULE_ITEM_NMB400_GROUP_ID_BERNOULLI = dritem_nmb400_b_20_21-09-23
LEAKY_ITEM_NMB400_GROUP_ID_BERNOULLI = leakyitem_nmb400_b_20_21-09-23
GRU_NMB400_GROUP_ID_BERNOULLI = gru11_nmb400_b_2021-09-23
LSTM_NMB400_GROUP_ID_BERNOULLI = lstm11_nmb400_b_2021-09-28
LSTM_GROUP_ID_MARKOV_INDEPENDENT = lstm11_mi_2021-09-28

define modelsdir_from_groupid
$(MODELS_PARENT_DIR)/$(1)
endef

# ENABLE_DEPENDENCY_ON_MODELS_DIR = TRUE
define dependency_modelsdir_if_enabled_from_groupid
$(if $(ENABLE_DEPENDENCY_ON_MODELS_DIR),$(call modelsdir_from_groupid,$(1)))
endef

DATASET_SUFFIX = _test_dict.pt
dataset_path_from_name = $(DATASET_DIR)/$(1)$(DATASET_SUFFIX)

################################################################################
# CONVENIENCE TARGETS
################################################################################

.PHONY: analyses_trained_agents
## Run all the code needed to reproduce the results
## of the analyses of trained agents reported in the paper
## (in Figure 2 to 7 and their Figure supplements and in the text of the corresponding sections),
## using the trained agent models located in $(MODELS_PARENT_DIR).
analyses_trained_agents: performance \
	prediction_sequence \
	learning_rate \
	readout \
	dynamics \
	perturbation_experiment \
	higher_level_inference

.PHONY: clean_analyses_trained_agents
## Delete all files generated by analyses_trained_agents
clean_analyses_trained_agents: clean_performance \
	clean_prediction_sequence \
	clean_learning_rate \
	clean_readout \
	clean_dynamics \
	clean_perturbation_experiment \
	clean_higher_level_inference

################################################################################
# TRAINING AGENTS WORKFLOWS
################################################################################

TRAINING_ALL_GROUP_IDS = $(sort \
	$(MAIN_GROUP_IDS_BERNOULLI) \
	$(MAIN_GROUP_IDS_MARKOV_INDEPENDENT) \
	$(MAIN_GROUP_IDS_MARKOV_COUPLED) \
	$(NETWORK_GROUP_IDS_PER_VOLATILITY) \
	$(NETWORK_GROUP_ID_BERNOULLI_THROUGH_TRAINING) \
	$(GRURESERVOIR_GROUP_ID_BERNOULLI) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_COUPLED) \
	$(ELMAN_GROUP_ID_BERNOULLI) \
	$(ELMAN_GROUP_ID_MARKOV_INDEPENDENT) \
	$(ELMAN_GROUP_ID_MARKOV_COUPLED) \
	$(GRUDIAG_GROUP_ID_BERNOULLI) \
	$(GRUDIAG_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRUDIAG_GROUP_ID_MARKOV_COUPLED) \
	$(DELTARULE_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(LEAKY_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(GRU_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_GROUP_ID_MARKOV_INDEPENDENT) \
	)
	# sort to remove duplicates

# Run training with given hyperparam config using ray tune API,
# saving the raw ray data temporarily,
# then extract what's relevant from the ray data
# to create the final trained models' group directory and delete the raw ray data

hyperparamconfig_from_groupid = $(HYPERPARAMCONFIG_DIR)/$(1).txt
raydata_from_groupid = $(RAY_DIR)/$(1)
TRAINING_RAYDATA_SRC_NOTDIR = training_run_config.py
TRAINING_MODELSDIR_SRC_NOTDIR = training_extract_data.py
TRAINING_RAYDATA_SRC = $(SRC_DIR)/$(TRAINING_RAYDATA_SRC_NOTDIR)
TRAINING_MODELSDIR_SRC = $(SRC_DIR)/$(TRAINING_MODELSDIR_SRC_NOTDIR)

define rule_modelsdir_with_hyperparamconfig_groupid
$(1): $(2) $(TRAINING_RAYDATA_SRC) $(TRAINING_MODELSDIR_SRC)
	$(eval TMP_RAYDATA=$(call raydata_from_groupid,$(3)))
	python $(TRAINING_RAYDATA_SRC) $(2) -o $(TMP_RAYDATA)
	python $(TRAINING_MODELSDIR_SRC) $(TMP_RAYDATA) $(3) -o $(1)
	rm -rf $(TMP_RAYDATA)
endef

$(foreach groupid,$(TRAINING_ALL_GROUP_IDS),$\
	$(eval $(call rule_modelsdir_with_hyperparamconfig_groupid,$\
		$(call modelsdir_from_groupid,$(groupid)),$\
		$(call hyperparamconfig_from_groupid,$(groupid)),$\
		$(groupid))))

# Plot training progress

trainingplotpng_from_groupid = $(TRAININGPLOT_DIR)/$(1)_training_progress.png
TRAININGPLOT_SRC_NOTDIR = training_plot_progress.py
TRAININGPLOT_SRC = $(SRC_DIR)/$(TRAININGPLOT_SRC_NOTDIR)

define rule_trainingplot_with_groupid
$(1): $(call dependency_modelsdir_if_enabled_from_groupid,$(2)) $(TRAININGPLOT_SRC)
	python $(TRAININGPLOT_SRC) $(2) -o $(1)
endef

$(foreach groupid,$(TRAINING_ALL_GROUP_IDS),$\
	$(eval $(call rule_trainingplot_with_groupid,$\
		$(call trainingplotpng_from_groupid,$(groupid)),$\
		$(groupid))))

trainingplotpngsmall_from_groupid = $(TRAININGPLOT_DIR)/$(1)_training_progress_small.png
TRAININGPLOT_SMALL_WIDTH = 2.2
TRAININGPLOT_SMALL_HEIGHT = 2.0
GRU_TRAININGPLOT_SMALL_GROUP_IDS = $(GRU_GROUP_ID_BERNOULLI) $(GRU_GROUP_ID_MARKOV_INDEPENDENT)
$(foreach groupid, $(GRU_TRAININGPLOT_SMALL_GROUP_IDS),$\
	$(eval GRU_TRAININGPLOT_PNGS_SMALL += $(call trainingplotpngsmall_from_groupid,$(groupid))))
$(foreach groupid, $(GRU_TRAININGPLOT_SMALL_GROUP_IDS),$\
	$(eval $(call rule_trainingplot_with_groupid,$\
		$(call trainingplotpngsmall_from_groupid,$(groupid)),$\
		$(groupid)) --width $(TRAININGPLOT_SMALL_WIDTH) --height $(TRAININGPLOT_SMALL_HEIGHT)))

# Phony targets

TRAINING_AGENTS_GROUP_IDS = $(sort \
	$(MAIN_GROUP_IDS_BERNOULLI) \
	$(MAIN_GROUP_IDS_MARKOV_INDEPENDENT) \
	$(MAIN_GROUP_IDS_MARKOV_COUPLED) \
	$(NETWORK_GROUP_IDS_PER_VOLATILITY) \
	$(GRURESERVOIR_GROUP_ID_BERNOULLI) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_COUPLED) \
	$(ELMAN_GROUP_ID_BERNOULLI) \
	$(ELMAN_GROUP_ID_MARKOV_INDEPENDENT) \
	$(ELMAN_GROUP_ID_MARKOV_COUPLED) \
	$(GRUDIAG_GROUP_ID_BERNOULLI) \
	$(GRUDIAG_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRUDIAG_GROUP_ID_MARKOV_COUPLED) \
	$(DELTARULE_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(LEAKY_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(GRU_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_GROUP_ID_MARKOV_INDEPENDENT) \
	)
	# sort to remove duplicates
TRAINING_AGENTS_MODELSDIRS = $(foreach groupid,$(TRAINING_AGENTS_GROUP_IDS),\
	$(call modelsdir_from_groupid,$(groupid)))

.PHONY: training_agents
training_agents: $(TRAINING_AGENTS_MODELSDIRS) \
	$(GRU_TRAININGPLOT_PNGS_SMALL)

.PHONY: clean_training_agents
clean_training_agents:
	rm -rf $(MODELS_PARENT_DIR)/*
	rm -rf $(TRAININGPLOT_DIR)/*

TRAINING_ALL_MODELSDIRS = $(foreach groupid,$(TRAINING_ALL_GROUP_IDS),\
	$(call modelsdir_from_groupid,$(groupid)))
TRAINING_ALL_TRAININGPLOT_PNGS = $(foreach groupid,$(TRAINING_ALL_GROUP_IDS),\
	$(call trainingplotpng_from_groupid,$(groupid)))

.PHONY: training_all
training_all: $(TRAINING_ALL_MODELSDIRS) $(TRAINING_ALL_TRAININGPLOT_PNGS) \
	$(GRU_TRAININGPLOT_PNGS_SMALL)

.PHONY: clean_training_all
clean_training_all:
	rm -f $(TRAINING_ALL_TRAININGPLOT_PNGS)
	rm -f $(GRU_TRAININGPLOT_PNGS_SMALL)
	rm -rf $(TRAINING_ALL_MODELSDIRS)

################################################################################
# HYPERPARAMETER OPTIMIZATION WORKFLOWS
################################################################################

ID_GRIDSEARCHUNITTYPENUNIT_BERNOULLI = gridsearch-unittype-nunit_b_20_20-09-6
ID_GRIDSEARCHUNITTYPENUNIT_MARKOV_INDEPENDENT = gridsearch-unittype-nunit_mi_20_20-09-6
ID_GRIDSEARCHUNITTYPENUNIT_MARKOV_COUPLED = gridsearch-unittype-nunit_mc_20_20-09-6

# Run grid search training with given hyperparam config using ray tune API and save raw ray data

define rule_raydata_with_hyperparamconfig
$(1): $(2) $(TRAINING_RAYDATA_SRC)
	python $(TRAINING_RAYDATA_SRC) $(2) -o $(1)
endef

TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval HYPERPARAMCONFIG_GRIDSEARCHUNITTYPENUNIT_$(task) = \
		$(call hyperparamconfig_from_groupid,$(ID_GRIDSEARCHUNITTYPENUNIT_$(task)))))
$(foreach task,$(TASK_KEYS),\
	$(eval RAYDATA_GRIDSEARCHUNITTYPENUNIT_$(task) = \
		$(call raydata_from_groupid,$(ID_GRIDSEARCHUNITTYPENUNIT_$(task)))))
$(foreach task,$(TASK_KEYS),\
	$(eval $(call rule_raydata_with_hyperparamconfig,$\
	$(RAYDATA_GRIDSEARCHUNITTYPENUNIT_$(task)),$\
	$(HYPERPARAMCONFIG_GRIDSEARCHUNITTYPENUNIT_$(task)))))

# Run hyperparameter optimization for each network type, number of units and task
# using Bayesian optimization to sample hyperparameter values,
# and select the set of hyperparameter values that has yielded the best results.

HYPERPARAM_OPT_TRIAL_DATA_SRC_NOTDIR = hyperparam_opt_trial_data.py
HYPERPARAM_OPT_AGGREGATE_DATA_SRC_NOTDIR = hyperparam_opt_aggregate_data.py
HYPERPARAM_OPT_SELECT_SRC_NOTDIR = hyperparam_opt_select.py
HYPERPARAM_OPT_TRIAL_DATA_SRC = $(SRC_DIR)/$(HYPERPARAM_OPT_TRIAL_DATA_SRC_NOTDIR)
HYPERPARAM_OPT_AGGREGATE_DATA_SRC = $(SRC_DIR)/$(HYPERPARAM_OPT_AGGREGATE_DATA_SRC_NOTDIR)
HYPERPARAM_OPT_SELECT_SRC = $(SRC_DIR)/$(HYPERPARAM_OPT_SELECT_SRC_NOTDIR)

ID_HYPERPARAMOPT_GRU11_B = skopt-ucb_adamlr-initwihstd-initwhhstd_gru11_b_20-12-10
ID_HYPERPARAMOPT_GRU45_B = skopt-ucb_adamlr-initwihstd-initwhhstd_gru45_b_20-12-10
ID_HYPERPARAMOPT_GRU03_B = skopt-ucb_adamlr-initwihstd-initwhhstd_gru3_b_20-12-12
ID_HYPERPARAMOPT_ELMAN11_B = skopt-ucb_adamlr-initwihstd-initwhhstd_elman11_b_20-12-10
ID_HYPERPARAMOPT_ELMAN45_B = skopt-ucb_adamlr-initwihstd-initwhhstd_elman45_b_20-12-10
ID_HYPERPARAMOPT_ELMAN03_B = skopt-ucb_adamlr-initwihstd-initwhhstd_elman3_b_20-12-12
ID_HYPERPARAMOPT_ELMAN1000_B = skopt-ucb_adamlr-initwihstd-initwhhstd_elman1000_b_21-09-27
ID_HYPERPARAMOPT_GRUDIAG11_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag11_b_20-12-10
ID_HYPERPARAMOPT_GRUDIAG45_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag45_b_20-12-10
ID_HYPERPARAMOPT_GRUDIAG03_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag3_b_20-12-12
ID_HYPERPARAMOPT_GRURESERVOIR11_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir11_b_20-12-10
ID_HYPERPARAMOPT_GRURESERVOIR45_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir45_b_20-12-10
ID_HYPERPARAMOPT_GRURESERVOIR03_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir3_b_20-12-12
ID_HYPERPARAMOPT_LSTM11_NMB400_B = skopt-ucb_adamlr-initwihstd-initwhhstd_lstm11_b_21-09-28
ID_HYPERPARAMOPT_GRU11_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_gru11_mi_20-12-10
ID_HYPERPARAMOPT_GRU45_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_gru45_mi_20-12-10
ID_HYPERPARAMOPT_GRU03_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_gru3_mi_20-12-12
ID_HYPERPARAMOPT_ELMAN11_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_elman11_mi_20-12-10
ID_HYPERPARAMOPT_ELMAN45_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_elman45_mi_20-12-10
ID_HYPERPARAMOPT_ELMAN03_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_elman3_mi_20-12-12
ID_HYPERPARAMOPT_ELMAN1000_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_elman1000_mi_21-09-27
ID_HYPERPARAMOPT_GRUDIAG11_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag11_mi_20-12-10
ID_HYPERPARAMOPT_GRUDIAG45_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag45_mi_20-12-10
ID_HYPERPARAMOPT_GRUDIAG03_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grudiag3_mi_20-12-12
ID_HYPERPARAMOPT_GRURESERVOIR11_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir11_mi_20-12-10
ID_HYPERPARAMOPT_GRURESERVOIR45_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir45_mi_20-12-10
ID_HYPERPARAMOPT_GRURESERVOIR03_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir3_mi_20-12-12
ID_HYPERPARAMOPT_GRURESERVOIR474_B = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir474_b_20-12-13
ID_HYPERPARAMOPT_GRURESERVOIR474_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_grureservoir474_mi_20-12-13
ID_HYPERPARAMOPT_LSTM11_MI = skopt-ucb_adamlr-initwihstd-initwhhstd_lstm11_mi_21-09-28

define rule_trialdata_with_raydata
$(1): $(2) $(HYPERPARAM_OPT_TRIAL_DATA_SRC)
	rm -f $(2)/experiment_state-*.json
	python $(HYPERPARAM_OPT_TRIAL_DATA_SRC) $(2) -o $(1)
endef

define rule_aggregatedata_with_trialdata
$(1): $(2) $(HYPERPARAM_OPT_AGGREGATE_DATA_SRC)
	python $(HYPERPARAM_OPT_AGGREGATE_DATA_SRC) $(2) -o $(1)
endef

define rule_select_with_aggregatedata_mode
$(1): $(2) $(HYPERPARAM_OPT_SELECT_SRC)
	python $(HYPERPARAM_OPT_SELECT_SRC) $(2) --mode $(3) -o $(1)
endef

KEYS = GRU11_B \
	GRU45_B \
	GRU03_B \
	ELMAN11_B \
	ELMAN45_B \
	ELMAN03_B \
	GRUDIAG11_B \
	GRUDIAG45_B \
	GRUDIAG03_B \
	GRURESERVOIR11_B \
	GRURESERVOIR45_B \
	GRURESERVOIR03_B \
	GRU11_MI \
	GRU45_MI \
	GRU03_MI \
	ELMAN11_MI \
	ELMAN45_MI \
	ELMAN03_MI \
	GRUDIAG11_MI \
	GRUDIAG45_MI \
	GRUDIAG03_MI \
	GRURESERVOIR11_MI \
	GRURESERVOIR45_MI \
	GRURESERVOIR03_MI \
	ELMAN1000_B \
	ELMAN1000_MI \
	GRURESERVOIR474_B \
	GRURESERVOIR474_MI \
	LSTM11_NMB400_B \
	LSTM11_MI
$(foreach key,$(KEYS),\
	$(eval HYPERPARAMCONFIG_HYPERPARAMOPT_$(key) = \
		$(call hyperparamconfig_from_groupid,$(ID_HYPERPARAMOPT_$(key)))))
$(foreach key,$(KEYS),\
	$(eval RAYDATA_HYPERPARAMOPT_$(key) = \
		$(call raydata_from_groupid,$(ID_HYPERPARAMOPT_$(key)))))
$(foreach key,$(KEYS),\
	$(eval TRIALDATA_HYPERPARAMOPT_$(key) = \
		$(HYPERPARAM_OPT_TRIAL_DATA_SRC_NOTDIR:%.py=$(HYPERPARAM_DIR)/%_$(call lc,$(key)).csv)))
$(foreach key,$(KEYS),\
	$(eval AGGREGATEDATA_HYPERPARAMOPT_$(key) = \
		$(HYPERPARAM_OPT_AGGREGATE_DATA_SRC_NOTDIR:%.py=$(HYPERPARAM_DIR)/%_$(call lc,$(key)).csv)))
$(foreach key,$(KEYS),\
	$(eval SELECT_MEAN_HYPERPARAMOPT_$(key) = \
		$(HYPERPARAM_OPT_SELECT_SRC_NOTDIR:%.py=$(HYPERPARAM_DIR)/%_mean_$(call lc,$(key)).csv)))
$(foreach key,$(KEYS),\
	$(eval SELECT_MEAN_HYPERPARAMOPT_ALL += $(SELECT_MEAN_HYPERPARAMOPT_$(key))))
$(foreach key,$(KEYS),\
	$(eval $(call rule_raydata_with_hyperparamconfig,$\
		$(RAYDATA_HYPERPARAMOPT_$(key)),$\
		$(HYPERPARAMCONFIG_HYPERPARAMOPT_$(key)))))
$(foreach key,$(KEYS),\
	$(eval $(call rule_trialdata_with_raydata,$\
		$(TRIALDATA_HYPERPARAMOPT_$(key)),$\
		$(RAYDATA_HYPERPARAMOPT_$(key)))))
$(foreach key,$(KEYS),\
	$(eval $(call rule_aggregatedata_with_trialdata,$\
		$(AGGREGATEDATA_HYPERPARAMOPT_$(key)),$\
		$(TRIALDATA_HYPERPARAMOPT_$(key)))))
$(foreach key,$(KEYS),\
	$(eval $(call rule_select_with_aggregatedata_mode,$\
		$(SELECT_MEAN_HYPERPARAMOPT_$(key)),$\
		$(AGGREGATEDATA_HYPERPARAMOPT_$(key)),mean)))

KEYS = ELMAN1000_B \
	ELMAN1000_MI
$(foreach key,$(KEYS),\
	$(eval SELECT_MIN_HYPERPARAMOPT_$(key) = \
		$(HYPERPARAM_OPT_SELECT_SRC_NOTDIR:%.py=$(HYPERPARAM_DIR)/%_min_$(call lc,$(key)).csv)))
$(foreach key,$(KEYS),\
	$(eval SELECT_MIN_HYPERPARAMOPT_ALL += $(SELECT_MIN_HYPERPARAMOPT_$(key))))
$(foreach key,$(KEYS),\
	$(eval $(call rule_select_with_aggregatedata_mode,$\
		$(SELECT_MIN_HYPERPARAMOPT_$(key)),$\
		$(AGGREGATEDATA_HYPERPARAMOPT_$(key)),min)))

.PHONY: hyperparam_opt
hyperparam_opt: $(SELECT_MEAN_HYPERPARAMOPT_ALL) $(SELECT_MIN_HYPERPARAMOPT_ALL)

################################################################################
# COMPLEXITY WORKFLOWS
################################################################################

# Plot performance over number of units for different network architectures

ID_GRID_NUNIT_GRU_BERNOULLI = grid_nunit_gru_b_2020-12-12
ID_GRID_NUNIT_ELMAN_BERNOULLI = grid_nunit_elman_b_2020-12-12
ID_GRID_NUNIT_LAST_ELMAN_BERNOULLI = grid_nunit_last_elman_b_2021-09-27
ID_GRID_NUNIT_GRUDIAG_BERNOULLI = grid_nunit_grudiag_b_2020-12-12
ID_GRID_NUNIT_GRURESERVOIR_BERNOULLI = grid_nunit_grureservoir_b_2020-12-12
ID_GRID_NUNIT_LAST_GRURESERVOIR_BERNOULLI = grid_nunit_last_grureservoir_b_2020-12-12
ID_GRID_NUNIT_GRU_MARKOV_INDEPENDENT = grid_nunit_gru_mi_2020-12-12
ID_GRID_NUNIT_ELMAN_MARKOV_INDEPENDENT = grid_nunit_elman_mi_2020-12-12
ID_GRID_NUNIT_LAST_ELMAN_MARKOV_INDEPENDENT = grid_nunit_last_elman_mi_2021-09-27
ID_GRID_NUNIT_GRUDIAG_MARKOV_INDEPENDENT = grid_nunit_grudiag_mi_2020-12-12
ID_GRID_NUNIT_GRURESERVOIR_MARKOV_INDEPENDENT = grid_nunit_grureservoir_mi_2020-12-12
ID_GRID_NUNIT_LAST_GRURESERVOIR_MARKOV_INDEPENDENT = grid_nunit_last_grureservoir_mi_2020-12-12

HYPERPARAM_COMPARE_NUNIT_DATA_SRC_NOTDIR = hyperparam_compare_nunit_data.py
HYPERPARAM_COMPARE_NUNIT_DATA_SRC = $(SRC_DIR)/$(HYPERPARAM_COMPARE_NUNIT_DATA_SRC_NOTDIR)
HYPERPARAM_COMPARE_NUNIT_PLOT_SRC_NOTDIR = hyperparam_compare_nunit_plot.py
HYPERPARAM_COMPARE_NUNIT_PLOT_SRC = $(SRC_DIR)/$(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC_NOTDIR)

define rule_compare_nunit_data_with_raydata
$(1): $(2) $(HYPERPARAM_COMPARE_NUNIT_DATA_SRC)
	python $(HYPERPARAM_COMPARE_NUNIT_DATA_SRC) $(2) -o $(1)
endef

define rule_compare_nunit_plot_with_data_style_legendstyle
$(1): $(2) $(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC)
	python $(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC) $(2) -o $(1) \
		--style $(3) --legend_style $(4)
endef

define rule_compare_nunit_plot_with_data_style_legendstyle_seconddata
$(1): $(2) $(5) $(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC)
	python $(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC) $(2) -o $(1) \
		--style $(3) --legend_style $(4) --second_data_path $(5)
endef

complexity_raydata_from_groupid = $(COMPLEXITY_RAY_DIR)/$(1)

# Make data files
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
ARCH_KEYS = GRU ELMAN GRUDIAG GRURESERVOIR LAST_ELMAN LAST_GRURESERVOIR
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval HYPERPARAMCONFIG_GRID_NUNIT_$(key)_$(task) = \
			$(call hyperparamconfig_from_groupid,$(ID_GRID_NUNIT_$(key)_$(task))))))
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval RAYDATA_GRID_NUNIT_$(key)_$(task) = \
			$(call complexity_raydata_from_groupid,$(ID_GRID_NUNIT_$(key)_$(task))))))
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval RAYDATA_GRID_NUNIT_ALL += $(RAYDATA_GRID_NUNIT_$(key)_$(task)))))
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval HYPERPARAM_COMPARE_NUNIT_DATAFILE_$(key)_$(task) = \
			$(HYPERPARAM_COMPARE_NUNIT_DATA_SRC_NOTDIR:%.py=$(COMPLEXITY_DIR)/%_$(ARCHNAME_$(key))_$(TASKNAMELC_$(task)).csv))))
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval $(call rule_raydata_with_hyperparamconfig,$\
			$(RAYDATA_GRID_NUNIT_$(key)_$(task)),$\
			$(HYPERPARAMCONFIG_GRID_NUNIT_$(key)_$(task))))))
$(foreach task, $(TASK_KEYS),\
	$(foreach key,$(ARCH_KEYS),\
		$(eval $(call rule_compare_nunit_data_with_raydata,$\
			$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_$(key)_$(task)),$\
			$(RAYDATA_GRID_NUNIT_$(key)_$(task))))))

# Make plots
IMG_TYPES = png svg
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
ARCH_KEYS = GRU ELMAN GRUDIAG GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task, $(TASK_KEYS),\
		$(foreach key,$(ARCH_KEYS),\
			$(eval HYPERPARAM_COMPARE_NUNIT_$(imgtype)_$(key)_$(task) = \
				$(HYPERPARAM_COMPARE_NUNIT_PLOT_SRC_NOTDIR:%.py=$(COMPLEXITY_DIR)/%_$(ARCHNAME_$(key))_$(TASKNAMELC_$(task)).$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task, $(TASK_KEYS),\
		$(foreach key,$(ARCH_KEYS),\
			$(eval HYPERPARAM_COMPARE_NUNIT_IMG_ALL += $(HYPERPARAM_COMPARE_NUNIT_$(imgtype)_$(key)_$(task))))))

IMG_TYPES = png svg
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
ARCH_KEYS = GRU GRUDIAG
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task, $(TASK_KEYS),\
		$(foreach key,$(ARCH_KEYS),\
			$(eval $(call rule_compare_nunit_plot_with_data_style_legendstyle,$\
				$(HYPERPARAM_COMPARE_NUNIT_$(imgtype)_$(key)_$(task)),$\
				$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_$(key)_$(task)),$\
				paper,mechanisms)))))

IMG_TYPES = png svg
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
ARCH_KEYS = ELMAN GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task, $(TASK_KEYS),\
		$(foreach key,$(ARCH_KEYS),\
			$(eval $(call rule_compare_nunit_plot_with_data_style_legendstyle_seconddata,$\
					$(HYPERPARAM_COMPARE_NUNIT_$(imgtype)_$(key)_$(task)),$\
					$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_$(key)_$(task)),$\
					paper,mechanisms,$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_LAST_$(key)_$(task)))))))

HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC_NOTDIR = hyperparam_nunit_fit_power_law.py
HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC = $(SRC_DIR)/$(HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC_NOTDIR)
HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT = \
	$(HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC_NOTDIR:%.py=$(COMPLEXITY_DIR)/%_$(ARCHNAME_ELMAN)_$(TASKNAMELC_BERNOULLI).txt)

$(HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT): $(HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC) \
	$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_ELMAN_BERNOULLI) \
	$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_LAST_ELMAN_BERNOULLI)
	python $(HYPERPARAM_NUNIT_FIT_POWER_LAW_SRC) \
		$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_ELMAN_BERNOULLI) \
		$(HYPERPARAM_COMPARE_NUNIT_DATAFILE_LAST_ELMAN_BERNOULLI) \
		-o $(HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT)

# Phony targets

COMPLEXITY_IMGS = $(HYPERPARAM_COMPARE_NUNIT_IMG_ALL)
COMPLEXITY_POWER_LAW_OUTPUT = $(HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT)

.PHONY: complexity
complexity: $(COMPLEXITY_IMGS) \
	$(COMPLEXITY_POWER_LAW_OUTPUT)

.PHONY: clean_complexity
clean_complexity:
	rm -rf $(COMPLEXITY_DIR)/*
	rm -rf $(COMPLEXITY_RAY_DIR)/*

.PHONY: complexity_compare_nunit
complexity_compare_nunit: \
	$(HYPERPARAM_COMPARE_NUNIT_IMG_ALL)

.PHONY: complexity_nunit_fit_power_law
complexity_nunit_fit_power_law: \
	$(HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT)

.PHONY: clean_complexity_compare_nunit
clean_complexity_compare_nunit:
	rm -f $(HYPERPARAM_COMPARE_NUNIT_IMG_ALL)
	rm -rf $(RAYDATA_GRID_NUNIT_ALL)

.PHONY: clean_complexity_nunit_fit_power_law
clean_complexity_nunit_fit_power_law:
	rm -f $(HYPERPARAM_NUNIT_FIT_POWER_LAW_OUTPUT)

################################################################################
# PERFORMANCE WORKFLOWS
################################################################################

DATAFILE_KWD = data
STATSFILE_KWD = stats

PERF_GROUP_IDS_BERNOULLI = $(HEURISTIC_GROUP_IDS_BERNOULLI) \
	$(HEURISTIC_GROUP_IDS_MARKOV_INDEPENDENT) $(HEURISTIC_GROUP_IDS_MARKOV_COUPLED) \
	$(ELMAN_GROUP_ID_BERNOULLI) \
	$(GRUDIAG_GROUP_ID_BERNOULLI) \
	$(GRURESERVOIR_GROUP_ID_BERNOULLI) \
	$(DELTARULE_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(LEAKY_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(GRU_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_NMB400_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $(NETWORK_GROUP_ID_MARKOV_COUPLED)
PERF_GROUP_IDS_MARKOV_INDEPENDENT = $(HEURISTIC_GROUP_IDS_BERNOULLI) \
	$(HEURISTIC_GROUP_IDS_MARKOV_INDEPENDENT) $(HEURISTIC_GROUP_IDS_MARKOV_COUPLED) \
	$(ELMAN_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRUDIAG_GROUP_ID_MARKOV_INDEPENDENT) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_INDEPENDENT) \
	$(LSTM_GROUP_ID_MARKOV_INDEPENDENT) \
	$(NETWORK_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $(NETWORK_GROUP_ID_MARKOV_COUPLED)
PERF_GROUP_IDS_MARKOV_COUPLED = $(HEURISTIC_GROUP_IDS_BERNOULLI) \
	$(HEURISTIC_GROUP_IDS_MARKOV_INDEPENDENT) $(HEURISTIC_GROUP_IDS_MARKOV_COUPLED) \
	$(ELMAN_GROUP_ID_MARKOV_COUPLED) \
	$(GRUDIAG_GROUP_ID_MARKOV_COUPLED) \
	$(GRURESERVOIR_GROUP_ID_MARKOV_COUPLED) \
	$(NETWORK_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $(NETWORK_GROUP_ID_MARKOV_COUPLED)

DATASET_NAME_TEST_PERF_BERNOULLI = B_PC1by75_N1000_06-03-20
DATASET_NAME_TEST_PERF_MARKOV_INDEPENDENT = MI_PC1by75_N1000_06-03-20
DATASET_NAME_TEST_PERF_MARKOV_COUPLED = MC_PC1by75_N1000_06-03-20

PERF_DATA_SRC_NOTDIR = performance_test_data.py
PERF_DATA_SRC = $(SRC_DIR)/$(PERF_DATA_SRC_NOTDIR)

define rule_perf_data_with_datasetname_groupids_outputdata_outputstats
$(3) $(4): $(PERF_DATA_SRC) \
	$(call dataset_path_from_name,$(1)) \
	$(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid)))
	python $(PERF_DATA_SRC) $(1) $(2) -o $(3) --do_output_stats -o-stats-path $(4)
endef

TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval PERF_DATAFILE_$(task) = \
		$(PERF_DIR)/$(PERF_DATA_SRC_NOTDIR:%_data.py=%_$(DATAFILE_KWD)_$(TASKNAME_$(task)).csv)))
$(foreach task,$(TASK_KEYS),\
	$(eval PERF_STATSFILE_$(task) = \
		$(PERF_DIR)/$(PERF_DATA_SRC_NOTDIR:%_data.py=%_$(STATSFILE_KWD)_$(TASKNAME_$(task)).csv)))
$(foreach task,$(TASK_KEYS),\
	$(eval $(call rule_perf_data_with_datasetname_groupids_outputdata_outputstats,$\
		$(DATASET_NAME_TEST_PERF_$(task)),$(PERF_GROUP_IDS_$(task)),$\
			$(PERF_DATAFILE_$(task)),$(PERF_STATSFILE_$(task)))))

PERF_PLOT_SRC_NOTDIR = performance_test_plot.py
PERF_PLOT_SRC = $(SRC_DIR)/$(PERF_PLOT_SRC_NOTDIR)

define rule_perf_plot_with_datafile_groupids_style_legendstyle
$(1): $(2) $(PERF_PLOT_SRC)
	python $(PERF_PLOT_SRC) $(2) $(3) -o $(1) --style $(4) --legend_style $(5)
endef

PERF_PLOT_MECHANISMS_BERNOULLI_GROUP_IDS = \
	$(DELTARULE_ITEM_GROUP_ID_BERNOULLI) $(LEAKY_ITEM_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_BERNOULLI) \
	$(ELMAN_GROUP_ID_BERNOULLI) \
	$(GRUDIAG_GROUP_ID_BERNOULLI) \
	$(GRURESERVOIR_GROUP_ID_BERNOULLI)
TASK_KEYS = MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach task,$(TASK_KEYS),\
	$(eval PERF_PLOT_MECHANISMS_$(task)_GROUP_IDS = \
		$(HEURISTIC_GROUP_IDS_$(task)) \
		$(NETWORK_GROUP_ID_$(task)) \
		$(ELMAN_GROUP_ID_$(task)) \
		$(GRUDIAG_GROUP_ID_$(task)) \
		$(GRURESERVOIR_GROUP_ID_$(task))))

PERF_PLOT_SMALL_WIDTH = 3.37
PERF_PLOT_SMALL_HEIGHT_BERNOULLI = 2.16
PERF_PLOT_SMALL_HEIGHT_MARKOV_INDEPENDENT = 3.07
PERF_PLOT_SMALL_HEIGHT_MARKOV_COUPLED = $(PERF_PLOT_SMALL_HEIGHT_MARKOV_INDEPENDENT)

IMG_TYPES = PNG SVG
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT MARKOV_COUPLED
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task,$(TASK_KEYS),\
		$(eval PERF_PLOT_$(imgtype)_MECHANISMS_$(task)_SMALL = \
			$(PERF_DIR)/$(PERF_PLOT_SRC_NOTDIR:%.py=%_$(TASKNAME_$(task))_heuristic-gru-elman-grudiag-grureservoir_smallsize.$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task,$(TASK_KEYS),\
		$(eval PERF_PLOT_IMG_MECHANISMS_$(task)_SMALL_ALL += $(PERF_PLOT_$(imgtype)_MECHANISMS_$(task)_SMALL))))
$(foreach task,$(TASK_KEYS),\
		$(eval PERF_PLOT_IMG_MECHANISMS_SMALL_ALL += $(PERF_PLOT_IMG_MECHANISMS_$(task)_SMALL_ALL)))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(TASK_KEYS),\
		$(eval $(call rule_perf_plot_with_datafile_groupids_style_legendstyle,$\
		$(PERF_PLOT_$(imgtype)_MECHANISMS_$(key)_SMALL),$\
		$(PERF_DATAFILE_$(key)),$(PERF_PLOT_MECHANISMS_$(key)_GROUP_IDS),paper,mechanisms) \
		--width $(PERF_PLOT_SMALL_WIDTH) --height $(PERF_PLOT_SMALL_HEIGHT_$(key)))))

PERF_PLOT_GATING_GROUP_IDS_BERNOULLI = \
	$(DELTARULE_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(LEAKY_ITEM_NMB400_GROUP_ID_BERNOULLI) \
	$(GRU_NMB400_GROUP_ID_BERNOULLI) \
	$(LSTM_NMB400_GROUP_ID_BERNOULLI)
PERF_PLOT_GATING_GROUP_IDS_MARKOV_INDEPENDENT = \
	$(HEURISTIC_GROUP_IDS_MARKOV_INDEPENDENT) \
	$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) \
	$(LSTM_GROUP_ID_MARKOV_INDEPENDENT)

IMG_TYPES = png svg
TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task,$(TASK_KEYS),\
		$(eval PERF_PLOT_$(imgtype)_GATING_$(task) = \
			$(PERF_DIR)/$(PERF_PLOT_SRC_NOTDIR:%.py=%_$(TASKNAME_$(task))_heuristic-gru-lstm_smallsize.$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task,$(TASK_KEYS),\
		$(eval PERF_PLOT_IMG_GATING_ALL += $(PERF_PLOT_$(imgtype)_GATING_$(task)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach task,$(TASK_KEYS),\
		$(eval $(call rule_perf_plot_with_datafile_groupids_style_legendstyle,$\
			$(PERF_PLOT_$(imgtype)_GATING_$(task)),$\
			$(PERF_DATAFILE_$(task)),$(PERF_PLOT_GATING_GROUP_IDS_$(task)),paper,gating) \
			--width $(PERF_PLOT_SMALL_WIDTH) --height $(PERF_PLOT_SMALL_HEIGHT_$(task)))))

PERF_TEST_COMPARISON_SRC_NOTDIR = performance_test_comparison_stat.py
PERF_TEST_COMPARISON_SRC = $(SRC_DIR)/$(PERF_TEST_COMPARISON_SRC_NOTDIR)

define rule_perf_comparison_with_datafile_groupids
$(1): $(2) $(PERF_TEST_COMPARISON_SRC)
	python $(PERF_TEST_COMPARISON_SRC) $(2) $(3) -o $(1)
endef

TASK_KEYS = BERNOULLI MARKOV_INDEPENDENT
$(foreach key,$(TASK_KEYS),\
	$(eval PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_$(key) = \
		$(PERF_DIR)/$(PERF_TEST_COMPARISON_SRC_NOTDIR:%.py=%_$(TASKNAME_$(key))_heuristic-gru-elman-grudiag-grureservoir.csv)))
$(foreach key,$(TASK_KEYS),\
	$(eval PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_ALL += $(PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_$(key))))
$(foreach key,$(TASK_KEYS),\
	$(eval $(call rule_perf_comparison_with_datafile_groupids,$\
	$(PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_$(key)),$\
	$(PERF_DATAFILE_$(key)),$(PERF_PLOT_MECHANISMS_$(key)_GROUP_IDS))))

PERF_STATS_ANOVA_SRC_NOTDIR = performance_stats_anova_architecture_task.py
PERF_STATS_ANOVA_SRC = $(SRC_DIR)/$(PERF_STATS_ANOVA_SRC_NOTDIR)

define rule_perf_anova_with_datafiles_groupids_architectures_tasks
$(1): $(2) $(PERF_STATS_ANOVA_SRC)
	python $(PERF_STATS_ANOVA_SRC) \
		--data_paths $(2) \
		--group_ids $(3) \
		--architectures $(4) \
		--tasks $(5) \
		-o $(1)
endef

PERF_ANOVA_DATAFILES = $(PERF_DATAFILE_BERNOULLI) $(PERF_DATAFILE_BERNOULLI) \
	$(PERF_DATAFILE_MARKOV_INDEPENDENT) $(PERF_DATAFILE_MARKOV_INDEPENDENT)
PERF_ANOVA_TASKS = "unigram" "unigram" "bigram independent" "bigram independent"
ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR 
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERF_ANOVA_GRU_vs_$(key)_STATSFILE = \
		$(PERF_DIR)/$(PERF_STATS_ANOVA_SRC_NOTDIR:%_architecture_task.py=%_gru-vs-$(ARCHNAME_$(key))_unigram-vs-bigram.csv)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERF_ANOVA_STATSFILES += $(PERF_ANOVA_GRU_vs_$(key)_STATSFILE)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval $(call rule_perf_anova_with_datafiles_groupids_architectures_tasks,$\
	$(PERF_ANOVA_GRU_vs_$(key)_STATSFILE),$\
	$(PERF_ANOVA_DATAFILES),$\
	$(NETWORK_GROUP_ID_BERNOULLI) $($(key)_GROUP_ID_BERNOULLI) $(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $($(key)_GROUP_ID_MARKOV_INDEPENDENT),$\
	gru $(call lc,$(key)) gru $(call lc,$(key)),$\
	$(PERF_ANOVA_TASKS))))

PERF_PLOT_ACROSS_TRAIN_TEST_SRC_NOTDIR = performance_test_plot_across_train_test.py
PERF_PLOT_ACROSS_TRAIN_TEST_SRC = $(SRC_DIR)/$(PERF_PLOT_ACROSS_TRAIN_TEST_SRC_NOTDIR)
PERF_PLOT_ACROSS_TRAIN_TEST_DATAFILES = $(PERF_DATAFILE_BERNOULLI) \
	$(PERF_DATAFILE_MARKOV_INDEPENDENT) $(PERF_DATAFILE_MARKOV_COUPLED)

define rule_perf_plot_across_train_test_with_groupids_style_legendstyle
$(1): $(PERF_PLOT_ACROSS_TRAIN_TEST_DATAFILES) $(PERF_PLOT_ACROSS_TRAIN_TEST_SRC)
	python $(PERF_PLOT_ACROSS_TRAIN_TEST_SRC) --group_ids $(2) --data_paths $(PERF_PLOT_ACROSS_TRAIN_TEST_DATAFILES) -o $(1) --style $(3) --legend_style $(4)
endef

PERF_PLOT_ACROSS_TRAIN_TEST_GROUP_IDS_DELTARULE_GRU = \
	$(DELTARULE_ITEM_GROUP_ID_BERNOULLI) $(DELTARULE_ITEM_GROUP_ID_MARKOV_INDEPENDENT) $(DELTARULE_ITEM_GROUP_ID_MARKOV_COUPLED) \
	$(DELTARULE_TRANSITION_GROUP_ID_BERNOULLI) $(DELTARULE_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT) $(DELTARULE_TRANSITION_GROUP_ID_MARKOV_COUPLED) \
	$(NETWORK_GROUP_ID_BERNOULLI) $(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $(NETWORK_GROUP_ID_MARKOV_COUPLED)
PERF_PLOT_ACROSS_TRAIN_TEST_GROUP_IDS_LEAKY_GRU = \
	$(LEAKY_ITEM_GROUP_ID_BERNOULLI) $(LEAKY_ITEM_GROUP_ID_MARKOV_INDEPENDENT) $(LEAKY_ITEM_GROUP_ID_MARKOV_COUPLED) \
	$(LEAKY_TRANSITION_GROUP_ID_BERNOULLI) $(LEAKY_TRANSITION_GROUP_ID_MARKOV_INDEPENDENT) $(LEAKY_TRANSITION_GROUP_ID_MARKOV_COUPLED) \
	$(NETWORK_GROUP_ID_BERNOULLI) $(NETWORK_GROUP_ID_MARKOV_INDEPENDENT) $(NETWORK_GROUP_ID_MARKOV_COUPLED)

KEYS = DELTARULE_GRU LEAKY_GRU
IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(KEYS),\
		$(eval PERF_PLOT_ACROSS_TRAIN_TEST_$(key)_$(imgtype) = \
			$(PERF_DIR)/$(PERF_PLOT_ACROSS_TRAIN_TEST_SRC_NOTDIR:%.py=%_$(call lc,$(key)).$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(KEYS),\
		$(eval PERF_PLOT_PNG_ACROSS_TRAIN_TEST_ALL_IMGS += $(PERF_PLOT_ACROSS_TRAIN_TEST_$(key)_$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(KEYS),\
		$(eval $(call rule_perf_plot_across_train_test_with_groupids_style_legendstyle,$\
			$(PERF_PLOT_ACROSS_TRAIN_TEST_$(key)_$(imgtype)),$\
			$(PERF_PLOT_ACROSS_TRAIN_TEST_GROUP_IDS_$(key)),paper,training))))

PERFORMANCE_IMGS += $(PERF_PLOT_IMG_MECHANISMS_BERNOULLI_SMALL_ALL) \
	$(PERF_PLOT_IMG_MECHANISMS_MARKOV_INDEPENDENT_SMALL_ALL) \
	$(PERF_PLOT_PNG_ACROSS_TRAIN_TEST_ALL_IMGS) \
	$(PERF_PLOT_IMG_GATING_ALL)
PERFORMANCE_STATSFILES = $(PERF_STATSFILE_BERNOULLI) \
	$(PERF_STATSFILE_MARKOV_INDEPENDENT) \
	$(PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_ALL) \
	$(PERF_ANOVA_STATSFILES)

.PHONY: performance
performance: $(PERFORMANCE_IMGS) $(PERFORMANCE_STATSFILES)

.PHONY: clean_performance
clean_performance:
	rm -rf $(PERF_DIR)/*

.PHONY: performance_all
## Test performance of all models on all tasks
## and plot the model comparison figure
performance_all: $(PERF_PLOT_IMG_MECHANISMS_SMALL_ALL) \
	$(PERF_PLOT_IMG_GATING_ALL) \
	$(PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_ALL) \
	$(PERF_ANOVA_STATSFILES) \
	$(PERF_PLOT_PNG_ACROSS_TRAIN_TEST_ALL_IMGS)

.PHONY: performance_across_train_test
performance_across_train_test: $(PERF_PLOT_PNG_ACROSS_TRAIN_TEST_ALL_IMGS)

.PHONY: clean_performance_all
## Delete all data and figure for performance tests
clean_performance_all:
	rm -f $(PERF_PLOT_IMG_MECHANISMS_SMALL_ALL)
	rm -f $(PERF_PLOT_PNG_ACROSS_TRAIN_TEST_ALL_IMGS)
	rm -f $(PERF_PLOT_IMG_GATING_ALL)
	rm -f $(PERF_TEST_COMPARISON_STATSFILE_MECHANISMS_ALL)
	rm -f $(PERF_ANOVA_STATSFILES)
	rm -f $(PERF_DATAFILE_BERNOULLI)
	rm -f $(PERF_DATAFILE_MARKOV_INDEPENDENT)
	rm -f $(PERF_DATAFILE_MARKOV_COUPLED)
	rm -f $(PERF_STATSFILE_BERNOULLI)
	rm -f $(PERF_STATSFILE_MARKOV_INDEPENDENT)
	rm -f $(PERF_STATSFILE_MARKOV_COUPLED)
	
################################################################################
# PREDICTION SEQUENCE WORKFLOWS
################################################################################

# Plot of example sequence predictions

DATASET_NAME_BEHAVIOR_SEQ = B_PC1by75_N1000_06-03-20
DATASET_PATH_BEHAVIOR_SEQ = $(call dataset_path_from_name,$(DATASET_NAME_BEHAVIOR_SEQ))
INDEX_BEHAVIOR_SEQ = 862
BEHAVIOR_SEQ_SRC_NOTDIR = behavior_sequence_predictions_plot.py
BEHAVIOR_SEQ_SRC = $(SRC_DIR)/$(BEHAVIOR_SEQ_SRC_NOTDIR)

define rule_behavior_seq_png_with_groupids_style_legendstyle
$(1): $(DATASET_PATH_BEHAVIOR_SEQ) \
	$(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) \
	$(BEHAVIOR_SEQ_SRC)
	python $(BEHAVIOR_SEQ_SRC) \
		$(DATASET_NAME_BEHAVIOR_SEQ) $(INDEX_BEHAVIOR_SEQ) $(2) \
		 $(PERF_STATSFILE_BERNOULLI) \
		--style $(3) --legend_style $(4) -o $(1)
endef

BEHAVIOR_SEQ_SMALL_WIDTH = 3.46
BEHAVIOR_SEQ_LARGE_WIDTH = 9.84

BEHAVIOR_SEQ_GROUPIDS = $(NETWORK_GROUP_ID_BERNOULLI) $(LEAKY_ITEM_GROUP_ID_BERNOULLI) $(DELTARULE_ITEM_GROUP_ID_BERNOULLI) 
BEHAVIOR_SEQ_PNG = \
	$(PRED_SEQ_DIR)/$(BEHAVIOR_SEQ_SRC_NOTDIR:%.py=%_$(DATASET_NAME_BEHAVIOR_SEQ)_$(INDEX_BEHAVIOR_SEQ).png)
BEHAVIOR_SEQ_IMG_UNIGRAM_ALL += $(BEHAVIOR_SEQ_PNG)
$(eval $(call rule_behavior_seq_png_with_groupids_style_legendstyle,$\
	$(BEHAVIOR_SEQ_PNG),$(BEHAVIOR_SEQ_GROUPIDS),paper,default))
IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_SEQ_$(imgtype)_SMALL = \
		$(PRED_SEQ_DIR)/$(BEHAVIOR_SEQ_SRC_NOTDIR:%.py=%_$(DATASET_NAME_BEHAVIOR_SEQ)_$(INDEX_BEHAVIOR_SEQ)_smallwidth.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_behavior_seq_png_with_groupids_style_legendstyle,$\
		$(BEHAVIOR_SEQ_$(imgtype)_SMALL),$(BEHAVIOR_SEQ_GROUPIDS),paper,default) --width $(BEHAVIOR_SEQ_SMALL_WIDTH)))
BEHAVIOR_SEQ_IMG_UNIGRAM_ALL += $(BEHAVIOR_SEQ_PNG_SMALL) $(BEHAVIOR_SEQ_SVG_SMALL)
BEHAVIOR_SEQ_SVG_LARGE = \
		$(PRED_SEQ_DIR)/$(BEHAVIOR_SEQ_SRC_NOTDIR:%.py=%_$(DATASET_NAME_BEHAVIOR_SEQ)_$(INDEX_BEHAVIOR_SEQ)_largewidth.svg)
BEHAVIOR_SEQ_IMG_UNIGRAM_ALL += $(BEHAVIOR_SEQ_SVG_LARGE)
$(eval $(call rule_behavior_seq_png_with_groupids_style_legendstyle,$\
		$(BEHAVIOR_SEQ_SVG_LARGE),$(BEHAVIOR_SEQ_GROUPIDS),paper,default) --width $(BEHAVIOR_SEQ_LARGE_WIDTH))

.PHONY: prediction_sequence_unigram_all
## Plot figure showing the predictions of all models on a sample sequence
## using the model instance that yields the median level of performance
## in performance tests
prediction_sequence_unigram_all: $(BEHAVIOR_SEQ_IMG_UNIGRAM_ALL)

.PHONY: clean_prediction_sequence_unigram_all
clean_prediction_sequence_unigram_all:
	rm -f $(BEHAVIOR_SEQ_IMG_UNIGRAM_ALL)

# Plot of example sequence predictions on bigram environment

BEHAVIOR_SEQ_BIGRAMS_SRC_NOTDIR = behavior_sequence_predictions_bigrams_plot.py
BEHAVIOR_SEQ_BIGRAMS_SRC = $(SRC_DIR)/$(BEHAVIOR_SEQ_BIGRAMS_SRC_NOTDIR)

define rule_behavior_seq_bigram_png_with_groupids_style_legendstyle
$(1): $(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid)))\
	$(BEHAVIOR_SEQ_BIGRAMS_SRC)
	python $(BEHAVIOR_SEQ_BIGRAMS_SRC) $(2) $(PERF_STATSFILE_MARKOV_INDEPENDENT) \
		--style $(3) --legend_style $(4) -o $(1)
endef

ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR GRU 
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval BEHAVIOR_SEQ_BIGRAMS_GROUPIDS += $($(key)_GROUP_ID_MARKOV_INDEPENDENT)))
IMG_TYPES = png svg
$(foreach type,$(IMG_TYPES),\
	$(eval BEHAVIOR_SEQ_BIGRAMS_$(type) = $(PRED_SEQ_DIR)/$(BEHAVIOR_SEQ_BIGRAMS_SRC_NOTDIR:%.py=%.$(type))))
$(foreach type,$(IMG_TYPES),\
	$(eval BEHAVIOR_SEQ_BIGRAMS_IMGS += $(BEHAVIOR_SEQ_BIGRAMS_$(type))))

$(foreach type,$(IMG_TYPES),\
	$(eval $(call rule_behavior_seq_bigram_png_with_groupids_style_legendstyle,$\
	$(BEHAVIOR_SEQ_BIGRAMS_$(type)),$(BEHAVIOR_SEQ_BIGRAMS_GROUPIDS),paper,mechanisms)))

.PHONY: prediction_sequence_bigrams
## Plot figure showing the predictions of all models on a bigram sequence
## using the model instance that yields the median level of performance
## in performance tests
prediction_sequence_bigrams: $(BEHAVIOR_SEQ_BIGRAMS_IMGS)

.PHONY: clean_prediction_sequence_bigrams
clean_prediction_sequence_bigrams:
	rm -f $(BEHAVIOR_SEQ_BIGRAMS_IMGS)

PRED_SEQ_IMGS = $(BEHAVIOR_SEQ_PNG_SMALL) $(BEHAVIOR_SEQ_SVG_SMALL) $(BEHAVIOR_SEQ_BIGRAMS_IMGS)

.PHONY: prediction_sequence
prediction_sequence: $(PRED_SEQ_IMGS)

.PHONY: clean_prediction_sequence
clean_prediction_sequence:
	rm -rf $(PRED_SEQ_DIR)/*

################################################################################
# LEARNING RATE WORKFLOWS
################################################################################

# Plot of learning rate over time

BEHAVIOR_LROVERTIME_DATA_SRC_NOTDIR = behavior_lr_over_time_data.py
BEHAVIOR_LROVERTIME_DATA_SRC = $(SRC_DIR)/$(BEHAVIOR_LROVERTIME_DATA_SRC_NOTDIR)

define rule_behavior_lrovertime_data_with_groupids
$(1): $(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) \
	$(BEHAVIOR_LROVERTIME_DATA_SRC)
	python $(BEHAVIOR_LROVERTIME_DATA_SRC) $(2) -o $(1)
endef

BEHAVIOR_LROVERTIME_DATA_GROUP_IDS = \
	$(DELTARULE_ITEM_GROUP_ID_BERNOULLI) $(LEAKY_ITEM_GROUP_ID_BERNOULLI) \
	$(NETWORK_GROUP_ID_BERNOULLI) $(ELMAN_GROUP_ID_BERNOULLI) \
	$(GRUDIAG_GROUP_ID_BERNOULLI) \
	$(GRURESERVOIR_GROUP_ID_BERNOULLI)
BEHAVIOR_LROVERTIME_DATAFILE = $(LR_DIR)/$(BEHAVIOR_LROVERTIME_DATA_SRC_NOTDIR:%.py=%.pt)
$(eval $(call rule_behavior_lrovertime_data_with_groupids,$\
	$(BEHAVIOR_LROVERTIME_DATAFILE),$(BEHAVIOR_LROVERTIME_DATA_GROUP_IDS)))

BEHAVIOR_LROVERTIME_PLOT_SRC_NOTDIR = behavior_lr_over_time_plot.py
BEHAVIOR_LROVERTIME_PLOT_SRC = $(SRC_DIR)/$(BEHAVIOR_LROVERTIME_PLOT_SRC_NOTDIR)

define rule_behavior_lrovertime_png_with_groupids_style_legendstyle
$(1): $(BEHAVIOR_LROVERTIME_DATAFILE) \
	$(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) \
	$(BEHAVIOR_LROVERTIME_PLOT_SRC)
	python $(BEHAVIOR_LROVERTIME_PLOT_SRC) $(BEHAVIOR_LROVERTIME_DATAFILE) $(2) \
		--style $(3) --legend_style $(4) -o $(1)
endef

BEHAVIOR_LROVERTIME_PNG_GROUP_IDS = \
	$(NETWORK_GROUP_ID_BERNOULLI) $(LEAKY_ITEM_GROUP_ID_BERNOULLI) $(DELTARULE_ITEM_GROUP_ID_BERNOULLI)
BEHAVIOR_LROVERTIME_PNG = $(LR_DIR)/$(BEHAVIOR_LROVERTIME_PLOT_SRC_NOTDIR:%.py=%.png)
$(eval $(call rule_behavior_lrovertime_png_with_groupids_style_legendstyle,$\
	$(BEHAVIOR_LROVERTIME_PNG),$(BEHAVIOR_LROVERTIME_PNG_GROUP_IDS),paper,default))

BEHAVIOR_LROVERTIME_WIDTH_SMALL = 3.46
BEHAVIOR_LROVERTIME_NTIMESTEPS_SMALL = 150
IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_LROVERTIME_$(imgtype)_SMALL = \
		$(LR_DIR)/$(BEHAVIOR_LROVERTIME_PLOT_SRC_NOTDIR:%.py=%_smallwidth.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_LROVERTIME_IMGS_SMALL += $(BEHAVIOR_LROVERTIME_$(imgtype)_SMALL)))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_behavior_lrovertime_png_with_groupids_style_legendstyle,$\
		$(BEHAVIOR_LROVERTIME_$(imgtype)_SMALL),$(BEHAVIOR_LROVERTIME_PNG_GROUP_IDS),paper,default) \
		--width $(BEHAVIOR_LROVERTIME_WIDTH_SMALL) --n_time_steps $(BEHAVIOR_LROVERTIME_NTIMESTEPS_SMALL)))

IMG_TYPES = PNG SVG
ARCHITECTURE_KEYS = GRU ELMAN GRUDIAG GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval BEHAVIOR_LROVERTIME_$(key)_$(imgtype)_SMALL = $(LR_DIR)/$(BEHAVIOR_LROVERTIME_PLOT_SRC_NOTDIR:%.py=%_$(call lc,$(key))_smallwidth.$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval BEHAVIOR_LROVERTIME_IMGS_SMALL += $(BEHAVIOR_LROVERTIME_$(key)_$(imgtype)_SMALL))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval $(call rule_behavior_lrovertime_png_with_groupids_style_legendstyle,$\
		$(BEHAVIOR_LROVERTIME_$(key)_$(imgtype)_SMALL),$($(key)_GROUP_ID_BERNOULLI),paper,mechanisms) \
		--width $(BEHAVIOR_LROVERTIME_WIDTH_SMALL) --n_time_steps $(BEHAVIOR_LROVERTIME_NTIMESTEPS_SMALL) --hide_io --hide_legend)))

BEHAVIOR_LROVERTIME_IMG_ALL = $(BEHAVIOR_LROVERTIME_PNG) \
	$(BEHAVIOR_LROVERTIME_IMGS_SMALL)

.PHONY: lr_over_time
## Measure the modulations of learning rate over time
## for each type of model, and plot resulting figure
lr_over_time: $(BEHAVIOR_LROVERTIME_IMG_ALL)

.PHONY: clean_lr_over_time
## Delete all data and figure for learning rate over time
clean_lr_over_time:
	rm -f $(BEHAVIOR_LROVERTIME_IMG_ALL)
	rm -f $(BEHAVIOR_LROVERTIME_DATAFILE)

# Plots of learning rate across training and testing volatility

BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC_NOTDIR = behavior_lr_across_volatility_plot.py
BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC = $(SRC_DIR)/$(BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC_NOTDIR)
BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC_NOTDIR = behavior_lr_across_volatility_data.py
BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC = $(SRC_DIR)/$(BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC_NOTDIR)
BEHAVIOR_LRACROSSVOLATILITY_PNG_NETWORKS = \
	$(LR_DIR)/$(BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC_NOTDIR:%.py=%_networks.png)
BEHAVIOR_LRACROSSVOLATILITY_PNG_BAYES = \
	$(LR_DIR)/$(BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC_NOTDIR:%.py=%_bayes.png)
BEHAVIOR_LRACROSSVOLATILITY_PNGS = $(BEHAVIOR_LRACROSSVOLATILITY_PNG_NETWORKS) \
	$(BEHAVIOR_LRACROSSVOLATILITY_PNG_BAYES)
BEHAVIOR_LRACROSSVOLATILITY_DATAFILE = \
	$(LR_DIR)/$(BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC_NOTDIR:%.py=%.pt)

DATASET_NAMES_PER_VOLATILITY = \
	B_PC1by300_N1000_22-03-20 \
    B_PC1by150_N1000_09-03-20 \
    B_PC1by75_N1000_06-03-20 \
    B_PC2by75_N1000_09-03-20

DATASET_PATHS_PER_VOLATILITY = $(foreach name,$(DATASET_NAMES_PER_VOLATILITY),\
    $(call dataset_path_from_name,$(name)) \
)

BEHAVIOR_LRACROSSVOLATILITY_PNGS_RECIPE = \
	python $(BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC) \
	$(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE) \
	-o_networks $(BEHAVIOR_LRACROSSVOLATILITY_PNG_NETWORKS) \
	-o_bayes $(BEHAVIOR_LRACROSSVOLATILITY_PNG_BAYES)

$(BEHAVIOR_LRACROSSVOLATILITY_PNGS): $(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE) \
	$(BEHAVIOR_LRACROSSVOLATILITY_PLOT_SRC)
	$(BEHAVIOR_LRACROSSVOLATILITY_PNGS_RECIPE)

BEHAVIOR_LRACROSSVOLATILITY_DATAFILE_RECIPE = \
	python $(BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC) \
	$(DATASET_NAMES_PER_VOLATILITY) \
	--group_ids $(NETWORK_GROUP_IDS_PER_VOLATILITY) \
	 -o $(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE)

DEPENDENCIES_IF_ENABLED_MODEL_DIRS_PER_VOLATILITY = $(foreach groupid,$(NETWORK_GROUP_IDS_PER_VOLATILITY),\
	$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid)))

$(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE): $(DATASET_PATHS_PER_VOLATILITY) \
	$(NETWORK_MODEL_DIRS_PER_VOLATILITY) $(BEHAVIOR_LRACROSSVOLATILITY_DATA_SRC)
	$(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE_RECIPE)

.PHONY: behavior_lr_across_volatility
## Measure networks' and Bayes's average learning rate
## as a function of training/prior volatility and test volatility,
## and plot the resulting matrix figure
behavior_lr_across_volatility: $(BEHAVIOR_LRACROSSVOLATILITY_PNGS)

.PHONY: clean_behavior_lr_across_volatility
## Delete all data and figure for learning rate across training/prior and test volatility
clean_behavior_lr_across_volatility:
	rm -f $(BEHAVIOR_LRACROSSVOLATILITY_PNGS)
	rm -f $(BEHAVIOR_LRACROSSVOLATILITY_DATAFILE)

LEARNING_RATE_IMGS += $(BEHAVIOR_LROVERTIME_IMGS_SMALL)
LEARNING_RATE_IMGS += $(BEHAVIOR_LRACROSSVOLATILITY_PNGS)

.PHONY: learning_rate
learning_rate: $(LEARNING_RATE_IMGS)

.PHONY: clean_learning_rate
clean_learning_rate:
	rm -rf $(LR_DIR)/*

################################################################################
# HIGHER-LEVEL INFERENCE (COUPLING) WORKFLOWS
################################################################################

BEHAVIOR_TESTCOUPLING_DATA_SRC_NOTDIR = behavior_test_coupling_data.py
BEHAVIOR_TESTCOUPLING_DATA_SRC = $(SRC_DIR)/$(BEHAVIOR_TESTCOUPLING_DATA_SRC_NOTDIR)

define rule_behavior_testcoupling_datafile_with_groupidcoup_groupidind
$(1): $(BEHAVIOR_TESTCOUPLING_DATA_SRC) \
	$(call dependency_modelsdir_if_enabled_from_groupid,$(2)) $(call dependency_modelsdir_if_enabled_from_groupid,$(3))
	python $(BEHAVIOR_TESTCOUPLING_DATA_SRC) $(2) $(3) -o $(1)
endef

BEHAVIOR_TESTCOUPLING_DATAFILE = \
	$(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_DATA_SRC_NOTDIR:%.py=%.csv)
$(eval $(call rule_behavior_testcoupling_datafile_with_groupidcoup_groupidind,$\
	$(BEHAVIOR_TESTCOUPLING_DATAFILE),$(NETWORK_GROUP_ID_MARKOV_COUPLED),$\
		$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT)))

BEHAVIOR_TESTCOUPLING_DATAFILE_IDEALOBSERVER = \
	$(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_DATA_SRC_NOTDIR:%.py=%_idealobserver.csv)
$(eval $(call rule_behavior_testcoupling_datafile_with_groupidcoup_groupidind,$\
	$(BEHAVIOR_TESTCOUPLING_DATAFILE_IDEALOBSERVER),$(NETWORK_GROUP_ID_MARKOV_COUPLED),$\
		$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT)) --test_ideal_observer)

BEHAVIOR_TESTCOUPLING_DATAFILES += $(BEHAVIOR_TESTCOUPLING_DATAFILE) $(BEHAVIOR_TESTCOUPLING_DATAFILE_IDEALOBSERVER)

MODEL_KEYS = LEAKY_TRANSITION ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(MODEL_KEYS),\
	$(eval BEHAVIOR_TESTCOUPLING_DATAFILE_$(key) = $(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_DATA_SRC_NOTDIR:%.py=%_$(call lc,$(key)).csv)))
$(foreach key,$(MODEL_KEYS),\
	$(eval BEHAVIOR_TESTCOUPLING_DATAFILES += $(BEHAVIOR_TESTCOUPLING_DATAFILE_$(key))))
$(foreach key,$(MODEL_KEYS),\
	$(eval $(call rule_behavior_testcoupling_datafile_with_groupidcoup_groupidind,$\
	$(BEHAVIOR_TESTCOUPLING_DATAFILE_$(key)),$\
	$($(key)_GROUP_ID_MARKOV_COUPLED),$\
	$($(key)_GROUP_ID_MARKOV_INDEPENDENT))))

BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC_NOTDIR = behavior_test_coupling_plot_diff.py
BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC = $(SRC_DIR)/$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC_NOTDIR)

define rule_behavior_testcoupling_diff_png_with_datafile
$(1): $(2) $(BEHAVIOR_TESTCOUPLING_DATAFILE_IDEALOBSERVER) \
	$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC)
	python $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC) \
		$(2) $(BEHAVIOR_TESTCOUPLING_DATAFILE_IDEALOBSERVER) \
		--style paper -o $(1)
endef

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype) = \
		$(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC_NOTDIR:%.py=%.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS += $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype))))
BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMG_ALL += $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS)
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_behavior_testcoupling_diff_png_with_datafile,$\
		$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)),$\
		$(BEHAVIOR_TESTCOUPLING_DATAFILE))))

IMG_TYPES = PNG
MODEL_KEYS = LEAKY_TRANSITION ELMAN GRUDIAG GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_$(key) = $(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC_NOTDIR:%.py=%_$(call lc,$(key)).$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMG_ALL += $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_$(key)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval $(call rule_behavior_testcoupling_diff_png_with_datafile,$\
		$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_$(key)),$\
		$(BEHAVIOR_TESTCOUPLING_DATAFILE_$(key))))))

BEHAVIOR_TESTCOUPLING_SMALL_WIDTH = 1.04
BEHAVIOR_TESTCOUPLING_SMALL_HEIGHT = 0.81
IMG_TYPES = PNG SVG
MODEL_KEYS = ELMAN GRUDIAG GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_SMALL_$(key) = $(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_SRC_NOTDIR:%.py=%_$(call lc,$(key))_small.$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS_SMALL += $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_SMALL_$(key)))))
BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMG_ALL += $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS_SMALL)
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(MODEL_KEYS),\
		$(eval $(call rule_behavior_testcoupling_diff_png_with_datafile,$\
		$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_$(imgtype)_SMALL_$(key)),$\
		$(BEHAVIOR_TESTCOUPLING_DATAFILE_$(key))) \
		--width $(BEHAVIOR_TESTCOUPLING_SMALL_WIDTH) --height $(BEHAVIOR_TESTCOUPLING_SMALL_HEIGHT))))

BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC_NOTDIR = behavior_test_coupling_sequence_plot.py
BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC = $(SRC_DIR)/$(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC_NOTDIR)

define rule_behavior_testcoupling_sequence_png_with_groupidcoup_groupidind
$(1): $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC) \
	$(call dependency_modelsdir_if_enabled_from_groupid,$(2)) $(call dependency_modelsdir_if_enabled_from_groupid,$(3)) \
	$(PERF_STATSFILE_MARKOV_COUPLED) $(PERF_STATSFILE_MARKOV_INDEPENDENT)
	python $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC) $(2) $(3) \
		$(PERF_STATSFILE_MARKOV_COUPLED) $(PERF_STATSFILE_MARKOV_INDEPENDENT) \
		--style paper \
		--width $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_WIDTH) \
		--height $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_HEIGHT) \
		-o $(1)
endef

BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_WIDTH = 2.92
BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_HEIGHT = 1.52
IMG_TYPES = png svg
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_$(imgtype) = \
		$(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_SRC_NOTDIR:%.py=%.$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_IMGS += $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_behavior_testcoupling_sequence_png_with_groupidcoup_groupidind,$\
		$(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_$(imgtype)),$\
		$(NETWORK_GROUP_ID_MARKOV_COUPLED),$\
		$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT))))

BEHAVIOR_TESTCOUPLING_ANOVA_SRC_NOTDIR = behavior_test_coupling_stats_anova_architecture.py
BEHAVIOR_TESTCOUPLING_ANOVA_SRC = $(SRC_DIR)/$(BEHAVIOR_TESTCOUPLING_ANOVA_SRC_NOTDIR)

define rule_behavior_testcoupling_anova_with_datafiles_architectures
$(1): $(2) $(BEHAVIOR_TESTCOUPLING_ANOVA_SRC)
	python $(BEHAVIOR_TESTCOUPLING_ANOVA_SRC) \
		--data_paths $(2) \
		--architectures $(3) \
		-o $(1)
endef

ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR 
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval BEHAVIOR_TESTCOUPLING_ANOVA_GRU_vs_$(key)_STATSFILE = \
		$(COUPLING_DIR)/$(BEHAVIOR_TESTCOUPLING_ANOVA_SRC_NOTDIR:%_architecture.py=%_gru-vs-$(call lc,$(key)).csv)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval BEHAVIOR_TESTCOUPLING_ANOVA_STATSFILES += $(BEHAVIOR_TESTCOUPLING_ANOVA_GRU_vs_$(key)_STATSFILE)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval $(call rule_behavior_testcoupling_anova_with_datafiles_architectures,$\
	$(BEHAVIOR_TESTCOUPLING_ANOVA_GRU_vs_$(key)_STATSFILE),$\
	$(BEHAVIOR_TESTCOUPLING_DATAFILE) $(BEHAVIOR_TESTCOUPLING_DATAFILE_$(key)),$\
	gru $(call lc,$(key)))))

COUPLING_IMGS = $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_IMGS) \
	$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS) \
	$(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMGS_SMALL)
COUPLING_STATSFILES = $(BEHAVIOR_TESTCOUPLING_ANOVA_STATSFILES)

.PHONY: higher_level_inference
higher_level_inference: $(COUPLING_IMGS) $(COUPLING_STATSFILES)

.PHONY: clean_higher_level_inference
clean_higher_level_inference:
	rm -rf $(COUPLING_DIR)/*

.PHONY: higher_level_inference_all
higher_level_inference_all: $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMG_ALL) \
	$(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_IMGS) \
	$(BEHAVIOR_TESTCOUPLING_ANOVA_STATSFILES)

.PHONY: clean_higher_level_inference_all
clean_higher_level_inference_all:
	rm -f $(BEHAVIOR_TESTCOUPLING_PLOT_DIFF_IMG_ALL)
	rm -f $(BEHAVIOR_TESTCOUPLING_SEQUENCE_PLOT_IMGS)
	rm -f $(BEHAVIOR_TESTCOUPLING_ANOVA_STATSFILES)
	rm -f $(BEHAVIOR_TESTCOUPLING_DATAFILES)

################################################################################
# READOUT (DECODING) WORKFLOWS
################################################################################

# DECODING BERNOULLI CONFIDENCE WORKFLOWS

# Train and test linear regression decoders

DECODING_LINREG_DATA_SRC_NOTDIR = decoding_net_to_io_regression_data.py
DECODING_LINREG_DATA_SRC = $(SRC_DIR)/$(DECODING_LINREG_DATA_SRC_NOTDIR)
DECODING_LINREG_STATS_SRC_NOTDIR = decoding_net_to_io_regression_stats.py
DECODING_LINREG_STATS_SRC = $(SRC_DIR)/$(DECODING_LINREG_STATS_SRC_NOTDIR)

DATASET_NAME_DECODING_LINREG_BERNOULLI = B_PC1by75_N1000_06-03-20
DECODERS_DIR_BERNOULLI = $(DECODERS_PARENT_DIR)/$(NETWORK_GROUP_ID_BERNOULLI)
DECODERS_DIR_BERNOULLI_GRU = $(DECODERS_DIR_BERNOULLI)

define rule_decodersdir_with_datasetname_groupid
$(1): $(call dataset_path_from_name,$(2)) $(call dependency_modelsdir_if_enabled_from_groupid,$(3)) \
	$(DECODING_LINREG_DATA_SRC) $(DECODING_LINREG_STATS_SRC)
	python $(DECODING_LINREG_DATA_SRC) $(2) $(3) -o_dir $(1)
	python $(DECODING_LINREG_STATS_SRC) $(1)
endef

$(eval $(call rule_decodersdir_with_datasetname_groupid,$\
	$(DECODERS_DIR_BERNOULLI),$\
	$(DATASET_NAME_DECODING_LINREG_BERNOULLI),$\
	$(NETWORK_GROUP_ID_BERNOULLI)))

NETWORKARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(NETWORKARCHITECTURE_KEYS),\
	$(eval DECODERS_DIR_BERNOULLI_$(key) = $(DECODERS_PARENT_DIR)/$($(key)_GROUP_ID_BERNOULLI)))
$(foreach key,$(NETWORKARCHITECTURE_KEYS),\
	$(eval DECODERS_DIRS_BERNOULLI += $(DECODERS_DIR_BERNOULLI_$(key))))
$(foreach key,$(NETWORKARCHITECTURE_KEYS),\
	$(eval $(call rule_decodersdir_with_datasetname_groupid,$\
	$(DECODERS_DIR_BERNOULLI_$(key)),$\
	$(DATASET_NAME_DECODING_LINREG_BERNOULLI),$\
	$($(key)_GROUP_ID_BERNOULLI))))

# Plot R2

DECODING_LINREG_BERNOULLI_R2_PLOT_SRC_NOTDIR = decoding_bernoulli_r2_confidence_plot.py
DECODING_LINREG_BERNOULLI_R2_PLOT_SRC = $(SRC_DIR)/$(DECODING_LINREG_BERNOULLI_R2_PLOT_SRC_NOTDIR)
DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC_NOTDIR = decoding_bernoulli_r2_logodds_plot.py
DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC = $(SRC_DIR)/$(DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC_NOTDIR)
DECODING_LINREG_BERNOULLI_R2_PNG = $(DECODING_DIR)/$(DECODING_LINREG_BERNOULLI_R2_PLOT_SRC_NOTDIR:%.py=%.png)
DECODING_LINREG_BERNOULLI_LOGODDS_R2_PNG = \
	$(DECODING_DIR)/$(DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC_NOTDIR:%.py=%.png)

define rule_decoding_bernoulli_r2_plot_with_decodersdir_output_style
$(2): $(1) $(DECODING_LINREG_BERNOULLI_R2_PLOT_SRC)
	python $(DECODING_LINREG_BERNOULLI_R2_PLOT_SRC) $(1) --style $(3) -o $(2)
endef

define rule_decoding_bernoulli_logodds_r2_plot_with_decodersdir_output_style
$(2): $(1) $(DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC)
	python $(DECODING_LINREG_BERNOULLI_LOGODDS_R2_PLOT_SRC) $(1) --style $(3) -o $(2)
endef

$(eval $(call rule_decoding_bernoulli_r2_plot_with_decodersdir_output_style,$\
	$(DECODERS_DIR_BERNOULLI),$\
	$(DECODING_LINREG_BERNOULLI_R2_PNG),paper))

$(eval $(call rule_decoding_bernoulli_logodds_r2_plot_with_decodersdir_output_style,$\
	$(DECODERS_DIR_BERNOULLI),$\
	$(DECODING_LINREG_BERNOULLI_LOGODDS_R2_PNG),paper))

# Plot example sequence of confidence and learning rate

DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC_NOTDIR = decoding_bernoulli_sequence_confidence_lr_plot.py
DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC_NOTDIR)
DATASET_NAME_DECODING_CONFIDENCE_OVER_TIME = B_PC1by75_N1000_06-03-20
INDEX_SEQ_DECODING_CONFIDENCE_OVER_TIME = 2
IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_$(imgtype) = \
		$(DECODING_DIR)/$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC_NOTDIR:%.py=%_$(DATASET_NAME_DECODING_CONFIDENCE_OVER_TIME)_$(INDEX_SEQ_DECODING_CONFIDENCE_OVER_TIME).$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_IMGS += \
			$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_$(imgtype))))

define rule_decoding_bernoulli_sequence_confidence_lr_plot_with_groupid_decoderdir_datasetname_iseq
$(1): $(call dependency_modelsdir_if_enabled_from_groupid,$(2)) $(3) $(call dataset_path_from_name,$(4)) \
	$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC)
	python $(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_SRC) $(2) $(3) \
		--dataset_name $(4) --i_sequence $(5) \
		--n_time_steps 200 --width 2.44 --style paper --hide_ylabel \
		-o $(1)
endef

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_decoding_bernoulli_sequence_confidence_lr_plot_with_groupid_decoderdir_datasetname_iseq,$\
		$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_$(imgtype)),$\
		$(NETWORK_GROUP_ID_BERNOULLI),$\
		$(DECODERS_DIR_BERNOULLI),$\
		$(DATASET_NAME_DECODING_CONFIDENCE_OVER_TIME),$\
		$(INDEX_SEQ_DECODING_CONFIDENCE_OVER_TIME))))

# Compute correlation of decoded confidence with subsequent learning rate

DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC_NOTDIR = decoding_bernoulli_confidence_accuracy_plot.py
DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC_NOTDIR)
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC_NOTDIR = decoding_bernoulli_correlate_confidence_lr_plot.py
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC_NOTDIR)
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC_NOTDIR = decoding_bernoulli_correlate_confidence_lr_data.py
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC_NOTDIR)
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATS_SRC_NOTDIR = decoding_bernoulli_correlate_confidence_lr_stats.py
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATS_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATS_SRC_NOTDIR)
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE = $(DECODING_DIR)/$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC_NOTDIR:%data.py=%data.csv)
DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATSFILE = $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE:%data.csv=%stats.csv)

DATASET_NAME_DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR = B_PC1by75_N300_21-03-20

ARCHITECTURE_KEYS = GRU ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_GROUP_IDS += $($(key)_GROUP_ID_BERNOULLI)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_DECODER_DIRS += $(DECODERS_DIR_BERNOULLI_$(key))))

DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_GROUP_IDS = $(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_GROUP_IDS)

ARCHITECTURE_KEYS = GRUDIAG ELMAN GRURESERVOIR GRU
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_GROUP_IDS += $($(key)_GROUP_ID_BERNOULLI)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_DECODER_DIRS += $(DECODERS_DIR_BERNOULLI_$(key))))

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_$(imgtype) = \
		$(DECODING_DIR)/$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC_NOTDIR:%.py=%_gru_elman_grudiag_grureservoir.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_IMGS += \
		$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_$(imgtype))))

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_$(imgtype) = \
		$(DECODING_DIR)/$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC_NOTDIR:%.py=%_gru_elman_grudiag_grureservoir.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_IMGS += \
		$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_$(imgtype))))

define rule_decoding_bernoulli_confidence_accuracy_plot_with_groupids_decoderdirs
$(1): $(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) $(3) \
	$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC)
	python $(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_SRC) $(3) --group_ids $(2) \
		-o $(1) --style paper --legend_style mechanisms
endef

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_decoding_bernoulli_confidence_accuracy_plot_with_groupids_decoderdirs,$\
		$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_$(imgtype)),$\
		$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_GROUP_IDS),$\
		$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_DECODER_DIRS))))

define rule_decoding_bernoulli_correlate_data_with_groupids_decoderdirs
$(1): $(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) $(3) \
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC)
	python $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_SRC) $(DATASET_NAME_DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR) \
		--group_ids $(2) --decoder_group_dirs $(3) -o $(1)
endef

$(eval $(call rule_decoding_bernoulli_correlate_data_with_groupids_decoderdirs,$\
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE),$\
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_GROUP_IDS),$\
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_DECODER_DIRS)))

$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATSFILE): \
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE) $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATS_SRC)
	python $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATS_SRC) $< -o $@

$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_IMGS): \
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE) $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC)
	python $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_SRC) $< $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_GROUP_IDS) \
		-o $@ --style paper --legend_style mechanisms

DECODING_BERNOULLI_DEPENDENCE_DATA_SRC_NOTDIR = decoding_bernoulli_dependence_confidence_prediction_data.py
DECODING_BERNOULLI_DEPENDENCE_DATA_SRC = $(SRC_DIR)/$(DECODING_BERNOULLI_DEPENDENCE_DATA_SRC_NOTDIR)
DECODING_BERNOULLI_DEPENDENCE_DATAFILE = $(DECODING_DIR)/$(DECODING_BERNOULLI_DEPENDENCE_DATA_SRC_NOTDIR:%.py=%.csv)
DATASET_NAME_DECODING_BERNOULLI_DEPENDENCE = $(DATASET_NAME_DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR)
DECODING_BERNOULLI_DEPENDENCE_DATA_GROUP_IDS = $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_GROUP_IDS)
DECODING_BERNOULLI_DEPENDENCE_DATA_DECODER_DIRS = $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATA_DECODER_DIRS)

define rule_decoding_bernoulli_dependence_data_with_groupids_decoderdirs
$(1): $(foreach groupid,$(2),$(call dependency_modelsdir_if_enabled_from_groupid,$(groupid))) $(3) \
	$(DECODING_BERNOULLI_DEPENDENCE_DATA_SRC)
	python $(DECODING_BERNOULLI_DEPENDENCE_DATA_SRC) $(DATASET_NAME_DECODING_BERNOULLI_DEPENDENCE) \
		--group_ids $(2) --decoder_group_dirs $(3) -o $(1)
endef

$(eval $(call rule_decoding_bernoulli_dependence_data_with_groupids_decoderdirs,$\
	$(DECODING_BERNOULLI_DEPENDENCE_DATAFILE),$\
	$(DECODING_BERNOULLI_DEPENDENCE_DATA_GROUP_IDS),$\
	$(DECODING_BERNOULLI_DEPENDENCE_DATA_DECODER_DIRS)))

.PHONY: decoding_bernoulli
## Decode Bayes's confidence from each network hidden layer on the Bernoulli task
## using linear regression,
## save the decoder models in a separate directory,
## plot a sample scatter plot (decoded vs Bayes confidence) with the median decoder,
## plot decoding performance of all decoders
decoding_bernoulli: $(DECODING_LINREG_BERNOULLI_R2_PNG) \
	$(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_IMGS) \
	$(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_IMGS) \
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_IMGS) \
	$(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATSFILE) \
	$(DECODING_BERNOULLI_DEPENDENCE_DATAFILE)

.PHONY: clean_decoding_bernoulli
## Delete all data and figures for decoding confidence on Bernoulli task
clean_decoding_bernoulli:
	rm -f $(DECODING_LINREG_BERNOULLI_R2_PNG)
	rm -f $(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_IMGS)
	rm -f $(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_IMGS)
	rm -f $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_IMGS)
	rm -f $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATSFILE)
	rm -f $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_DATAFILE)
	rm -f $(DECODING_BERNOULLI_DEPENDENCE_DATAFILE)
	rm -rf $(DECODERS_DIR_BERNOULLI)
	rm -rf $(DECODERS_DIRS_BERNOULLI)

# DECODING MARKOV TRANSITION PROBABILITIES WORKFLOWS

# Train and test linear regression decoders

DATASET_NAME_DECODING_LINREG_MARKOV_INDEPENDENT = MI_PC1by75_N1000_06-03-20
DATASET_NAME_DECODING_LINREG_MARKOV_COUPLED = MC_PC1by75_N1000_06-03-20

DECODERS_DIR_MARKOV_INDEPENDENT = $(DECODERS_PARENT_DIR)/$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT)
DECODERS_DIR_MARKOV_COUPLED = $(DECODERS_PARENT_DIR)/$(NETWORK_GROUP_ID_MARKOV_COUPLED)

$(eval $(call rule_decodersdir_with_datasetname_groupid,$\
	$(DECODERS_DIR_MARKOV_INDEPENDENT),$\
	$(DATASET_NAME_DECODING_LINREG_MARKOV_INDEPENDENT),$\
	$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT)))

$(eval $(call rule_decodersdir_with_datasetname_groupid,$\
	$(DECODERS_DIR_MARKOV_COUPLED),$\
	$(DATASET_NAME_DECODING_LINREG_MARKOV_COUPLED),$\
	$(NETWORK_GROUP_ID_MARKOV_COUPLED)))

# Plot R2

DECODING_LINREG_MARKOV_R2_PLOT_SRC_NOTDIR = decoding_markov_r2_logodds_confidence_plot.py
DECODING_LINREG_MARKOV_R2_PLOT_SRC = $(SRC_DIR)/$(DECODING_LINREG_MARKOV_R2_PLOT_SRC_NOTDIR)

DECODING_LINREG_MARKOV_INDEPENDENT_R2_LOGODDS_PNG = \
	$(DECODING_DIR)/decoding_markov_independent_r2_logodds_plot.png
DECODING_LINREG_MARKOV_INDEPENDENT_R2_CONFIDENCE_PNG = \
	$(DECODING_DIR)/decoding_markov_independent_r2_confidence_plot.png
DECODING_LINREG_MARKOV_COUPLED_R2_LOGODDS_PNG = \
	$(DECODING_DIR)/decoding_markov_coupled_r2_logodds_plot.png
DECODING_LINREG_MARKOV_COUPLED_R2_CONFIDENCE_PNG = \
	$(DECODING_DIR)/decoding_markov_coupled_r2_confidence_plot.png

define rule_decoding_markov_r2_plot_with_decoderdir_ologodds_oconfidence_style
$(2) $(3): $(1) $(DECODING_LINREG_MARKOV_R2_PLOT_SRC)
	python $(DECODING_LINREG_MARKOV_R2_PLOT_SRC) $(1) --style $(4) -o_logodds $(2) -o_confidence $(3)
endef

$(eval $(call rule_decoding_markov_r2_plot_with_decoderdir_ologodds_oconfidence_style,$\
	$(DECODERS_DIR_MARKOV_INDEPENDENT),$\
	$(DECODING_LINREG_MARKOV_INDEPENDENT_R2_LOGODDS_PNG),$\
	$(DECODING_LINREG_MARKOV_INDEPENDENT_R2_CONFIDENCE_PNG),paper))

$(eval $(call rule_decoding_markov_r2_plot_with_decoderdir_ologodds_oconfidence_style,$\
	$(DECODERS_DIR_MARKOV_COUPLED),$\
	$(DECODING_LINREG_MARKOV_COUPLED_R2_LOGODDS_PNG),$\
	$(DECODING_LINREG_MARKOV_COUPLED_R2_CONFIDENCE_PNG),paper))

# Plot sample decoded transition probabilities over time (Markov Independent)

DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC_NOTDIR = decoding_markov_sequence_bigrams_plot.py
DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC = $(SRC_DIR)/$(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC_NOTDIR)
IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_MARKOV_SEQUENCE_BIGRAMS_$(imgtype) = \
		$(DECODING_DIR)/$(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC_NOTDIR:%.py=%.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DECODING_MARKOV_SEQUENCE_BIGRAMS_IMGS += \
		$(DECODING_MARKOV_SEQUENCE_BIGRAMS_$(imgtype))))
DECODING_MARKOV_SEQUENCE_BIGRAMS_GRURESERVOIR_PNG_PRESENTATION = \
	$(DECODING_DIR)/$(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC_NOTDIR:%.py=%_grureservoir_presentation.png)
DECODING_MARKOV_SEQUENCE_BIGRAMS_BERNOULLI_TO_MARKOV_INDEPENDENT_PNG_PRESENTATION = \
	$(DECODING_DIR)/$(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC_NOTDIR:%.py=%_bernoulli_to_markov_independent_presentation.png)

define rule_decoding_transition_probabilities_with_groupid_decoderdir_output_style
$(3): $(call dependency_modelsdir_if_enabled_from_groupid,$(1)) $(2) $(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC)
	python $(DECODING_MARKOV_SEQUENCE_BIGRAMS_SRC) \
		$(1) $(2) --style $(4) -o $(3)
endef

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_decoding_transition_probabilities_with_groupid_decoderdir_output_style,$\
		$(NETWORK_GROUP_ID_MARKOV_INDEPENDENT),$\
		$(DECODERS_DIR_MARKOV_INDEPENDENT),$\
		$(DECODING_MARKOV_SEQUENCE_BIGRAMS_$(imgtype)),paper)))

.PHONY: decoding_markov
decoding_markov: \
	$(DECODING_LINREG_MARKOV_INDEPENDENT_R2_LOGODDS_PNG) \
	$(DECODING_LINREG_MARKOV_INDEPENDENT_R2_CONFIDENCE_PNG) \
	$(DECODING_LINREG_MARKOV_COUPLED_R2_LOGODDS_PNG) \
	$(DECODING_LINREG_MARKOV_COUPLED_R2_CONFIDENCE_PNG) \
	$(DECODING_MARKOV_SEQUENCE_BIGRAMS_IMGS)

.PHONY: clean_decoding_markov
clean_decoding_markov:
	rm -f $(DECODING_LINREG_MARKOV_INDEPENDENT_R2_LOGODDS_PNG)
	rm -f $(DECODING_LINREG_MARKOV_INDEPENDENT_R2_CONFIDENCE_PNG)
	rm -f $(DECODING_LINREG_MARKOV_COUPLED_R2_LOGODDS_PNG)
	rm -f $(DECODING_LINREG_MARKOV_COUPLED_R2_CONFIDENCE_PNG)
	rm -f $(DECODING_MARKOV_SEQUENCE_BIGRAMS_IMGS)
	rm -rf $(DECODERS_DIR_MARKOV_INDEPENDENT)
	rm -rf $(DECODERS_DIR_MARKOV_COUPLED)

READOUT_IMGS += $(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_IMGS)
READOUT_IMGS += $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_PLOT_IMGS)
READOUT_IMGS += $(DECODING_BERNOULLI_SEQUENCE_CONFIDENCE_LR_PLOT_IMGS)
READOUT_IMGS += $(DECODING_MARKOV_SEQUENCE_BIGRAMS_IMGS)
READOUT_STATSFILES += $(DECODING_BERNOULLI_CONFIDENCE_ACCURACY_PLOT_DECODER_DIRS)
READOUT_STATSFILES += $(DECODING_BERNOULLI_CORRELATE_CONFIDENCE_LR_STATSFILE)
READOUT_STATSFILES += $(DECODING_BERNOULLI_DEPENDENCE_DATAFILE)
READOUT_STATSFILES += $(DECODERS_DIR_MARKOV_INDEPENDENT)

.PHONY: readout
readout: $(READOUT_IMGS) $(READOUT_STATSFILES)

.PHONY: clean_readout
clean_readout:
	rm -rf $(DECODERS_PARENT_DIR)/*
	rm -rf $(DECODING_DIR)/*

################################################################################
# DYNAMICS WORKFLOWS
################################################################################

DYNAMICS_NETWORK_PLOT_SRC_NOTDIR = dynamics_network_prediction_precision_plot.py
DYNAMICS_NETWORK_PLOT_SRC = $(SRC_DIR)/$(DYNAMICS_NETWORK_PLOT_SRC_NOTDIR)
DYNAMICS_IO_PLOT_SRC_NOTDIR = dynamics_io_posterior_plot.py
DYNAMICS_IO_PLOT_SRC = $(SRC_DIR)/$(DYNAMICS_IO_PLOT_SRC_NOTDIR)
DYNAMICS_PLOT_WIDTH = 1.72
DYNAMICS_PLOT_HEIGHT = 1.72

define rule_dynamics_network_plot_with_groupid_decoderdir
$(1): $(call dependency_modelsdir_if_enabled_from_groupid,$(2)) $(3) $(DYNAMICS_NETWORK_PLOT_SRC)
	python $(DYNAMICS_NETWORK_PLOT_SRC) --width $(DYNAMICS_PLOT_WIDTH) --height $(DYNAMICS_PLOT_WIDTH) \
		--group_id $(2) --decoder_group_dir $(3) -o $(1)
endef

define rule_dynamics_io_plot_with_outputsingletime
$(1) $(2): $(DYNAMICS_IO_PLOT_SRC)
	python $(DYNAMICS_IO_PLOT_SRC) --width $(DYNAMICS_PLOT_WIDTH) --height $(DYNAMICS_PLOT_WIDTH) \
		-o $(1) --output_singletime $(2)
endef

ARCHITECTURE_KEYS = ELMAN GRU
IMG_TYPES = png svg
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval DYNAMICS_NETWORK_PLOT_$(key)_$(imgtype) = \
			$(DYNAMICS_DIR)/$(DYNAMICS_NETWORK_PLOT_SRC_NOTDIR:%.py=%_$(ARCHNAME_$(key)).$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval DYNAMICS_PLOT_ALL_IMGS += $(DYNAMICS_NETWORK_PLOT_$(key)_$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval $(call rule_dynamics_network_plot_with_groupid_decoderdir,$\
			$(DYNAMICS_NETWORK_PLOT_$(key)_$(imgtype)),$\
			$($(key)_GROUP_ID_BERNOULLI),$\
			$(DECODERS_DIR_BERNOULLI_$(key))))))

$(foreach imgtype,$(IMG_TYPES),\
	$(eval DYNAMICS_IO_PLOT_OVERTIME_$(imgtype) = \
		$(DYNAMICS_DIR)/$(DYNAMICS_IO_PLOT_SRC_NOTDIR:%.py=%_over_time.$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DYNAMICS_PLOT_ALL_IMGS += $(DYNAMICS_IO_PLOT_OVERTIME_$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DYNAMICS_IO_PLOT_SINGLETIME_$(imgtype) = \
		$(DYNAMICS_DIR)/$(DYNAMICS_IO_PLOT_SRC_NOTDIR:%.py=%_single_time.$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval DYNAMICS_PLOT_ALL_IMGS += $(DYNAMICS_IO_PLOT_SINGLETIME_$(imgtype))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_dynamics_io_plot_with_outputsingletime,$\
		$(DYNAMICS_IO_PLOT_OVERTIME_$(imgtype)),$\
		$(DYNAMICS_IO_PLOT_SINGLETIME_$(imgtype)))))

.PHONY: dynamics
dynamics: $(DYNAMICS_PLOT_ALL_IMGS)

.PHONY: clean_dynamics
clean_dynamics:
	rm -rf $(DYNAMICS_DIR)/*

################################################################################
# PERTURBATION EXPERIMENT WORKFLOWS
################################################################################

DATASET_NAME_PERTURBATION_CONFIDENCE = B_PC1by75_N300_21-03-20
PERTURBATION_CONFIDENCE_DATA_SRC_NOTDIR = perturbation_confidence_on_lr_data.py
PERTURBATION_CONFIDENCE_DATA_SRC = $(SRC_DIR)/$(PERTURBATION_CONFIDENCE_DATA_SRC_NOTDIR)

define rule_perturbation_confidence_datafile_with_datasetname_groupid_decoderdir
$(1): $(call dataset_path_from_name,$(2)) $(call dependency_modelsdir_if_enabled_from_groupid,$(3)) $(4) \
	$(PERTURBATION_CONFIDENCE_DATA_SRC)
	python $(PERTURBATION_CONFIDENCE_DATA_SRC) $(2) $(3) $(4) -o $(1)
endef

PERTURBATION_CONFIDENCE_DATAFILE = \
	$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_DATA_SRC_NOTDIR:%.py=%.npz)
$(eval $(call rule_perturbation_confidence_datafile_with_datasetname_groupid_decoderdir,$\
	$(PERTURBATION_CONFIDENCE_DATAFILE),$\
	$(DATASET_NAME_PERTURBATION_CONFIDENCE),$\
	$(NETWORK_GROUP_ID_BERNOULLI),$\
	$(DECODERS_DIR_BERNOULLI)))

ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_$(key)_DATAFILE = \
		$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_DATA_SRC_NOTDIR:%.py=%_$(ARCHNAME_$(key)).npz)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_DATAFILES += $(PERTURBATION_CONFIDENCE_$(key)_DATAFILE)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval $(call rule_perturbation_confidence_datafile_with_datasetname_groupid_decoderdir,$\
		$(PERTURBATION_CONFIDENCE_$(key)_DATAFILE),$\
		$(DATASET_NAME_PERTURBATION_CONFIDENCE),$\
		$($(key)_GROUP_ID_BERNOULLI),$\
		$(DECODERS_DIR_BERNOULLI_$(key)))))

PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC_NOTDIR = perturbation_confidence_on_lr_group_plot.py
PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC = $(SRC_DIR)/$(PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC_NOTDIR)

define rule_perturbation_confidence_plot_with_datafile_output_style
$(2): $(1) $(PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC)
	python $(PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC) $(1) --style $(3) -o $(2)
endef

IMG_TYPES = PNG SVG
$(foreach imgtype,$(IMG_TYPES),\
	$(eval PERTURBATION_CONFIDENCE_GROUP_PLOT_$(imgtype) = \
		$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC_NOTDIR:%.py=%.$(call lc,$(imgtype)))))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval $(call rule_perturbation_confidence_plot_with_datafile_output_style,$\
		$(PERTURBATION_CONFIDENCE_DATAFILE),$\
		$(PERTURBATION_CONFIDENCE_GROUP_PLOT_$(imgtype)),paper)))
$(foreach imgtype,$(IMG_TYPES),\
	$(eval PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS += $(PERTURBATION_CONFIDENCE_GROUP_PLOT_$(imgtype))))
PERTURBATION_CONFIDENCE_GROUP_PLOT_IMG_ALL += $(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS)

PERTURBATION_GROUP_PLOT_SMALL_WIDTH = 1.84
PERTURBATION_GROUP_PLOT_SMALL_HEIGHT = 0.77
IMG_TYPES = PNG SVG
ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval PERTURBATION_CONFIDENCE_$(key)_GROUP_PLOT_$(imgtype)_SMALL = \
			$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_GROUP_PLOT_SRC_NOTDIR:%.py=%_$(ARCHNAME_$(key))_small.$(call lc,$(imgtype))))))
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS_SMALL += $(PERTURBATION_CONFIDENCE_$(key)_GROUP_PLOT_$(imgtype)_SMALL))))
PERTURBATION_CONFIDENCE_GROUP_PLOT_IMG_ALL += $(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS_SMALL)
$(foreach imgtype,$(IMG_TYPES),\
	$(foreach key,$(ARCHITECTURE_KEYS),\
		$(eval $(call rule_perturbation_confidence_plot_with_datafile_output_style,$\
		$(PERTURBATION_CONFIDENCE_$(key)_DATAFILE),$\
		$(PERTURBATION_CONFIDENCE_$(key)_GROUP_PLOT_$(imgtype)_SMALL),paper) \
		--width $(PERTURBATION_GROUP_PLOT_SMALL_WIDTH) --height $(PERTURBATION_GROUP_PLOT_SMALL_HEIGHT))))

PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC_NOTDIR = perturbation_confidence_on_lr_slope_data.py
PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC = $(SRC_DIR)/$(PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC_NOTDIR)
PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC_NOTDIR = perturbation_confidence_on_lr_stats_test_slope.py
PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC = $(SRC_DIR)/$(PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC_NOTDIR)

define rule_perturbation_confidence_slopedatafile_with_datafile
$(1): $(2) $(PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC)
	python $(PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC) $(2) -o $(1)
endef

define rule_perturbation_confidence_stats_test_slope_with_slopedatafile1_slopedatafile2
$(1): $(2) $(3) $(PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC)
	python $(PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC) $(2) $(3) -o $(1)
endef

PERTURBATION_CONFIDENCE_GRU_DATAFILE = $(PERTURBATION_CONFIDENCE_DATAFILE)

ARCHITECTURE_KEYS = GRU ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_$(key)_SLOPE_DATAFILE = \
		$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_SLOPE_DATA_SRC_NOTDIR:%.py=%_$(ARCHNAME_$(key)).csv)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_STATSFILE_ALL += $(PERTURBATION_CONFIDENCE_$(key)_SLOPE_DATAFILE)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval $(call rule_perturbation_confidence_slopedatafile_with_datafile,$\
	$(PERTURBATION_CONFIDENCE_$(key)_SLOPE_DATAFILE),$\
	$(PERTURBATION_CONFIDENCE_$(key)_DATAFILE))))

ARCHITECTURE_KEYS = ELMAN GRUDIAG GRURESERVOIR
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_GRU_vs_$(key)_STATSFILE = \
		$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_STATS_TEST_SLOPE_SRC_NOTDIR:%.py=%_$(ARCHNAME_GRU)-vs-$(ARCHNAME_$(key)).csv)))
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval PERTURBATION_CONFIDENCE_TEST_SLOPE_STATSFILES += $(PERTURBATION_CONFIDENCE_GRU_vs_$(key)_STATSFILE)))
PERTURBATION_CONFIDENCE_STATSFILE_ALL += $(PERTURBATION_CONFIDENCE_TEST_SLOPE_STATSFILES)
$(foreach key,$(ARCHITECTURE_KEYS),\
	$(eval $(call rule_perturbation_confidence_stats_test_slope_with_slopedatafile1_slopedatafile2,$\
	$(PERTURBATION_CONFIDENCE_GRU_vs_$(key)_STATSFILE),$\
	$(PERTURBATION_CONFIDENCE_GRU_SLOPE_DATAFILE),$\
	$(PERTURBATION_CONFIDENCE_$(key)_SLOPE_DATAFILE))))

PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC_NOTDIR = perturbation_confidence_on_lr_individual_compound_plot.py
PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC = $(SRC_DIR)/$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC_NOTDIR)
PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_PNG = \
	$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC_NOTDIR:%.py=%.png)
PERTURBATION_CONFIDENCE_INDIVIDUAL_MEDIAN_PLOT_PNG = \
	$(PERTURBATION_DIR)/$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC_NOTDIR:%_compound_plot.py=%_median_plot.png)

PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_RECIPE = \
	python $(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC) \
	$(PERTURBATION_CONFIDENCE_DATAFILE) \
	-o $(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_PNG) \
	-o_median $(PERTURBATION_CONFIDENCE_INDIVIDUAL_MEDIAN_PLOT_PNG)

$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_PNG) \
$(PERTURBATION_CONFIDENCE_INDIVIDUAL_MEDIAN_PLOT_PNG): \
	$(PERTURBATION_CONFIDENCE_DATAFILE) $(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_SRC)
	$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_RECIPE)

PERTURBATION_IMGS = $(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS) \
	$(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMGS_SMALL)
PERTURBATION_STATSFILES = $(PERTURBATION_CONFIDENCE_TEST_SLOPE_STATSFILES)

.PHONY: perturbation_experiment
perturbation_experiment: $(PERTURBATION_IMGS) \
	$(PERTURBATION_STATSFILES)

.PHONY: clean_perturbation_experiment
clean_perturbation_experiment:
	rm -rf $(PERTURBATION_DIR)/*

.PHONY: perturbation_experiment_all
## Run an intervention experiment that systematically perturbs a network's
## representation of confidence and measure its effect on the network's
## subsequent learning rate,
## and plot figures showing the effect at the group level
## and at the individual level
perturbation_experiment_all: $(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMG_ALL)
	$(PERTURBATION_CONFIDENCE_STATSFILE_ALL) \
	$(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_PNG) \
	$(PERTURBATION_CONFIDENCE_INDIVIDUAL_MEDIAN_PLOT_PNG)

.PHONY: clean_perturbation_experiment_all
## Delete all data and figures for the intervention experiment
clean_perturbation_experiment_all:
	rm -f $(PERTURBATION_CONFIDENCE_GROUP_PLOT_IMG_ALL)
	rm -f $(PERTURBATION_CONFIDENCE_STATSFILE_ALL)
	rm -f $(PERTURBATION_CONFIDENCE_INDIVIDUAL_PLOT_PNG)
	rm -f $(PERTURBATION_CONFIDENCE_INDIVIDUAL_MEDIAN_PLOT_PNG)
	rm -f $(PERTURBATION_CONFIDENCE_DATAFILE)
	rm -f $(PERTURBATION_CONFIDENCE_DATAFILES)

################################################################################
# TRAINING_DYNAMICS WORKFLOWS
################################################################################

DATASET_NAME_TRAININGDYNAMICS_TRAIN = B_PC1by75_N1000_06-03-20
DATASET_NAME_TRAININGDYNAMICS_TEST = B_PC1by75_N1000_23-05-20
DATASET_PATH_TRAININGDYNAMICS_TRAIN = \
	$(call dataset_path_from_name,$(DATASET_NAME_TRAININGDYNAMICS_TRAIN))
DATASET_PATH_TRAININGDYNAMICS_TEST = \
	$(call dataset_path_from_name,$(DATASET_NAME_TRAININGDYNAMICS_TEST))
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC_NOTDIR = training_dynamics_prediction_confidence_data.py
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC = $(SRC_DIR)/$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC_NOTDIR)
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATAFILE = \
	$(TRAININGDYNAMICS_DIR)/$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC_NOTDIR:%.py=%.csv)

TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_RECIPE = \
	python $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC) \
	$(DATASET_NAME_TRAININGDYNAMICS_TRAIN) $(DATASET_NAME_TRAININGDYNAMICS_TEST) \
	$(NETWORK_GROUP_ID_BERNOULLI_THROUGH_TRAINING) \
	-o $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATAFILE)

$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATAFILE): \
	$(DATASET_PATH_TRAININGDYNAMICS_TRAIN) $(DATASET_PATH_TRAININGDYNAMICS_TEST) \
	$(call dependency_modelsdir_if_enabled_from_groupid,$(NETWORK_GROUP_ID_BERNOULLI_THROUGH_TRAINING)) \
	$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_SRC)
	$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATA_RECIPE)

TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC_NOTDIR = training_dynamics_prediction_confidence_plot.py
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC = $(SRC_DIR)/$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC_NOTDIR)

trainingdynamics_png_with_figtype = $(TRAININGDYNAMICS_DIR)/$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC_NOTDIR:%.py=%_$(1).png)
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_FIGTYPES = curve scatter trajectory performance
TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_PNGS = \
	$(foreach figtype,$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_FIGTYPES),\
	$(call trainingdynamics_png_with_figtype,$(figtype)))

define rule_trainingdynamics_prediction_confidence_plot_with_datafile_figtype
$(1): $(2) $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC)
	python $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_SRC) $(2) $(3) -o $(1)
endef

$(foreach figtype,$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_FIGTYPES),$\
	$(eval $(call rule_trainingdynamics_prediction_confidence_plot_with_datafile_figtype,$\
		$(call trainingdynamics_png_with_figtype,$(figtype)),$\
		$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATAFILE),$\
		$(figtype))))

.PHONY: training_dynamics
training_dynamics: \
	$(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_PNGS)

.PHONY: clean_training_dynamics
clean_training_dynamics:
	rm -f $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_PLOT_PNGS)
	rm -f $(TRAININGDYNAMICS_PREDICTION_CONFIDENCE_DATAFILE)
