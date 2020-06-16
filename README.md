# tensorflow_v2_cvae_sample

Sample implementation of Conditional Variational Autoencoder (CVAE).

<table>
  <tr>
    <td>sample-cvae-mnist.ipynb</td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1J1JDVBdc9v_WKcHDrTIhCEVjsF43NZ0T?usp=sharing">
        <img src="https://www.tensorflow.org/images/colab_logo_32px.png" width=20 />
        Google Colab</a>
    </td>
    <td>
        <a target="_blank" href="https://github.com/kn1cht/tensorflow_v2_cvae_sample/blob/master/sample-cvae-mnist.ipynb">
        <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" width=20 />
        GitHub</a>
    </td>
  </tr>
  <tr>
    <td>sample-cvae-mnist-manifold.ipynb</td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1q9UfVEADbecvejwKqWe59LDg2oSUiruz?usp=sharing">
        <img src="https://www.tensorflow.org/images/colab_logo_32px.png" width=20 />
        Google Colab</a>
    </td>
    <td>
        <a target="_blank" href="https://github.com/kn1cht/tensorflow_v2_cvae_sample/blob/master/sample-cvae-mnist-manifold.ipynb">
        <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" width=20 />
        GitHub</a>
    </td>
  </tr>
</table>

This code includes sample implementation from [TensorFlow Core Tutrial](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cvae.ipynb).

- Copyright 2020 The TensorFlow Authors. All rights reserved.
- distributed in the Apache License 2.0 (https://github.com/tensorflow/docs/blob/master/LICENSE)
- contributed by @lamberta @MarkDaoust @yashk2810
- changed by @kn1cht (June 15, 2020)

## Result
- After 100 epochs

![](images/image_at_epoch_0100.png)

- Continuously change writing style

![](images/result-writing-style.png)

- 2D manifold of the latent space

![](images/result-2d-manifold-2.png)![](images/result-2d-manifold-4.png)
