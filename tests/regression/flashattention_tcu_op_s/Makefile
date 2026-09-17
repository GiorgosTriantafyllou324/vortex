ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := flashattention_tcu_op_s

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(VORTEX_HOME)/tests/regression/flashattention_tcu_op/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

include ../common.mk
