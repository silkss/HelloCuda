# Простой проект для изучения CUDA
## Команда для комилирования
nvcc .\main.cu -arch=compute_50 --gpu_code=sm_50

- nvcc -- это компилятор CUDA.
- .\main.cu -- это исходный код, который компилируем.
- -arch=compute_50 -- указывает на *виртульную* архитектуру GPU.
- --gpu_code=sm_50 -- указывает на *реальную* архиктуру GPU.

**compute_50 и sm_50** это *виртуальная* и *реальная* архитектура **МОЕЙ** видеокарты (*GeForce 840M*)

