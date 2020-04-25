#
# GNU Make: targets that don't build files
#

.PHONY: all debug clean distclean

#
# Make targets
#

all: agents tests visualizer

debug: agents tests visualizer

agents:
	$(MAKE) -C jbw/agents $(MAKECMDGOALS)

agents_dbg:
	$(MAKE) -C jbw/agents $(MAKECMDGOALS)

tests:
	$(MAKE) -C jbw/tests $(MAKECMDGOALS)

tests_dbg:
	$(MAKE) -C jbw/tests $(MAKECMDGOALS)

visualizer:
	$(MAKE) -C jbw/visualizer $(MAKECMDGOALS)

visualizer_dbg:
	$(MAKE) -C jbw/visualizer $(MAKECMDGOALS)

clean: agents tests visualizer
