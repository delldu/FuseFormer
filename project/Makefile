all:
	@echo "Help"
	@echo "  make format"
	@echo "  make install"
	@echo "  make uninstall"
	@echo "  make clean"

format:
	@echo
	find . -name "*.py" -exec black -l 120 {} \;

install:
	python setup.py install

uninstall:
	pip uninstall video_patch

clean:
	rm -rf build dist video_patch.egg-info __pycache__
