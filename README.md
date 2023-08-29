# feature-matching

### 1. Find key points using the Harris corner detector. <br>
Given an image patch centered at $p$, the autocorrelation function is the 
difference between the original patch and a slightly shifted version of it centered
at $p'$. It is often accompanied by a window function that is often Gaussian 
giving more weight to pixels closer to $p$. Mathemetically, the autocorrelation 
function is expressed as

$$E_{AC}(\delta u) = \sum_i w(x_i)[I(x_i + \delta u) - I(x_i)]^2$$

where $I$ is the given image, $x_i$ is the image coordinate, $\Delta u$ is the displacement, and $i$ is the index for the coordinates in the patch. 

Using the first order Taylor expansion, we have $I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u.$

$$
\begin{equation}
\begin{split}
E_{AC}(\Delta u) &\approx \sum_i w(x_i)[I(x_i) + \nabla I(x_i) \cdot \Delta u - I(x_i)]^2 \\
                 &= \sum_i w(x_i)[\nabla I(x_i) \cdot \Delta u]^2  \\
                 &= \sum_i w(x_i)\Delta u^T A \Delta u.
\end{split}
\end{equation}
$$

