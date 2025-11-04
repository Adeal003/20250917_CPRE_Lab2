CprE 487/587
Lab 5: Multithreading, Tiling, and SIMD
“Some people, when confronted with a problem, think, “I know, I’ll use threads,” and
then two they hav erpoblesms.” –Ned Batchelder
1 Learning Objectives
By the end of this lab you should be able to
• Quantify the performance of a deep neural network classification model running on generalpurpose
hardware (CPUs).
• Evaluate software-based optimizations of a deep neural network classification models running
on a CPU leveraging hardware knowledge (data re-use from locality, SIMD execution, multithreaded
execution, and quantization).
2 Pre-lab
2.1 Lab Checklist
• Complete lab 2 (Or at least have a basic understanding of implementing neural network
layers).
• Have completed the C++ environment setup demonstrated in lab 2.
2.2 Intro
In this lab, we are going to be re-implementing and optimizing our previously implemented ML
model in C++. We will be exploring cache coherent operations an data access/reuse, SIMD operations,
and Multi-threading. Note: While we could also include quantization into this mix, including
quantization support for these implementations is OPTIONAL.
This lab should realistically take your group one week to complete.
