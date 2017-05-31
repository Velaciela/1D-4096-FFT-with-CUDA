# 1D-4096-FFT-with-CUDA

实测FFT算法在Maxwell架构上恰好处于计算密集和访存密集两类算法之间，

在做到足够优化的情况下，计算时间可以掩盖访存时间。

---

本项目使用Stockham结构实现并行FFT算法，达到与cuFFT一致的速度。

通过整合kernel，可实现比调用cuFFT更快的算法整体执行速度。

另外cuFFT分配了用户不可访问的显存空间，本项目避免了这一问题。

---

项目中测试了8192组4096点时域递增数的一维FFT计算。

结果保存于一个txt文件，可用MATLAB对比验证。

暂给出4096点FFT实现代码，文档请联系作者。

---

运行环境为WIN7 x64 + CUDA 7.5。


![image_1bbiue2qu1u58fd69dt1thb1ma88d.png-48.8kB][1]


  [1]: http://static.zybuluo.com/Velaciela/qatbq1qnvcjajx7tal9z26p5/image_1bbiue2qu1u58fd69dt1thb1ma88d.png
