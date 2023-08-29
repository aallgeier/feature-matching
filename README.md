# feature-matching

### 1. Find key points using the Harris corner detector. <br>
The auto correlation function tells us how much the pixel value changes at each 
location in a patch when we slightly shift it. It is accompanied by a window
function that is often Gaussian which gives more weight to pixels closer to the
patch center. 

Mathemetically, the autocorrelation function is expressed as
$$E_{AC}(\delta u) = \sum_{\text{patch}} w(x_i)[I(x_i + \Delta u) - I(x_i)]^2$$
where $I$ is the given image, $x_i$ is the 2D pixel coordinate, $\Delta u$ is the displacement.

If the patch contains a corner, the $E_{AC}(\delta u)$ will be larger in all directions.
If the patch contains an edge, $E_{AC}(\delta u)$ be large in one direction.
If the patch is flat $E_{AC}(\delta u)$ will be small.

Using the first order Taylor expansion, we have $I(x_i + \Delta u) \approx I(x_i) + \nabla I(x_i) \cdot \Delta u.$ Then the autocorrelation function can be approximated as

$$
\begin{equation}
\begin{split}
E_{AC}(\Delta u) &\approx \sum_i w(x_i)[I(x_i) + \nabla I(x_i) \cdot \Delta u - I(x_i)]^2 \\
                 &= \sum_i w(x_i)[\nabla I(x_i) \cdot \Delta u]^2  \\
                 &= \sum_i w(x_i)(I_x^2(x_i)\Delta x^2 + 2 I_xI_y(x_i)\Delta x \Delta y + I_y^2(x_i)\Delta y^2)\\
                 &= \sum_i w(x_i) (\Delta x \quad \Delta y)
                 \begin{pmatrix}
                 I_x(x_i)  & IxIy(x_i)\\
                 IxIy(x_i) & I_y(x_i)
                 \end{pmatrix}
                 \begin{pmatrix}
                 \Delta x\\
                 \Delta y
                 \end{pmatrix}
                 
                 &= \sum_i w(x_i)\Delta u^T A \Delta u
\end{split}
\end{equation}
$$

and $A$ is called the auto-correlation matrix. 


