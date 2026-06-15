# Planning

Project proposal and week-by-week development log.

- `aaidark1_project_proposal.pdf` — original project proposal

-------
## Week 1 Planning

### Goals ###
This is what you plan to accomplish in the first week of the project, leading up
to our lab meeting.  You must fill this in as part of your proposal.
* Hand in the full proposal and finalize the project timeline.
* Starting to dive deep into the Black-Scholes-Merton PDE to ensure the partial derivatives are correctly mapped for the PINN residual.
* Begin drafting the training loop and custom loss function in PyTorch or NVIDIA Modulus.
* Complete the foundational reading of the Raissi et al. (2019) paper to understand the collocation point sampling strategy.

### Update ###

* We handed in our proposal and created this timeline
* I have narrowed the project scope to Quantitative Finance, specifically using PINNs to price European and American options by solving the Black-Scholes-Merton equation.
* For literature review, I found the essential papers for PINNs and have started reading Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations (M. Raissi, P. Perdikaris, G. E. Karniadakis, 2019)
* Successfully set up NVIDIA SimNet (Modulus) on the Swarthmore Computer Science Society's NVIDIA RTX 6000. We have 96GB of VRAM at our disposal, which will allow me to handle significantly more collocation points than a standard consumer GPU.
* The setup for SimNet was quite temperamental; it broke multiple times during the environment configuration and dependency mapping, which required more time than initially planned.
* I am currently stuck on some of the more complex equations and inverse problem formulations within the Raissi paper.

### Requests ###
* I am meeting with Professor Pascal Kataboh to go over the specific differential equations to ensure my understanding of the math matches the implementation requirements. But other than these equations, everything seems good. 

------
## Week 2 Planning

This is what I plan to accomplish this week, moving from the 1D prototype to the full financial model.

### Goals ###
*  Expand the neural network input from a 1D spatial variable (x) to the 2D financial domain of Asset Price (S) and Time (t).
* Implement the "physics" residual for the Black-Scholes-Merton PDE.
* Aggregate the training metrics from the Poisson 1D baseline to demonstrate "Proof of Concept" during the lab presentation.
Hopefully, my implementation of PINNs on a simple Poisson baseline will prove the problem-method fit!

### Update ###

Complete this section ahead of lab for week 2 of the project
* Successfully implemented a 1D PINN to solve the Poisson equation d^2u/dx^2 = -sin(x). Verified the model converged to the exact solution u(x) = sin(x) using only the PDE and boundary conditions.
* Avoided LLM assistance to build the training loop, opting for the Deep Learning with PyTorch textbook. This ensured a fundamental understanding of how torch.autograd.grad handles second-order derivatives.
* Identified and resolved the versioning conflict on the RTX 6000 Blackwell. The environment mismatch between sm_120 (which is Blackwell family) and the local PyTorch (sm_80) build was causing kernel image errors. Switching to the latest Nightly build optimized for Blackwell fixed the throughput issues.
* Integrated the Alpha Vantage API. I can now pull live and historical options chains, which provides the real-world labels needed for the "Inverse Problem" part of the project.

### Requests ###
We need advice about/help with ... (optional):

------
## Week 3 Planning

This is what you plan to accomplish in the third week of the project, leading up
to our lab meeting.  You must fill this in as part of your proposal; update after
each checkpoint as necessary (i.e. as your plan evolves, this document should
reflect the changes). 

### Goals ###

* Transition the PyTorch code from the 1D Poisson setup to the 2D Black-Scholes domain (S, t).
* Train the model on synthetic data generated from the analytical Black-Scholes formula to verify that the PINN converges to the "ground truth."
* Experiment with different weightings (w_1, w_2, w_3) for the PDE and boundary losses

### Update ###

Complete this section ahead of lab for week 3 of the project

* I’ve successfully updated the architecture to handle two inputs (S and t) and mapped out the partial derivatives for the Black-Scholes residual using torch.autograd.
* The Alpha Vantage pipeline is fully functional, allowing me to pull real-world option chains to use as validation data.
* I'm running into some "gradient pathologies" where the PDE loss converges much faster than the boundary loss, causing the model to satisfy the math but miss the actual strike price payoff. I’m looking into adaptive loss weighting to fix this.

### Requests ###
No requests as of now.


------
## Week 4 Planning

This is what you plan to accomplish in the fourth week of the project, leading up
to our lab meeting.  You must fill this in as part of your proposal; update after
each checkpoint as necessary (i.e. as your plan evolves, this document should
reflect the changes). 

### Goals ###

* Transition from a fixed-strike model to a parametric solver that treats Strike Price (K) as a third input dimension (S, t, K).
* Derive the risk-free rate (r) and annualized historical volatility (\sigma) specifically for Apple Inc. (AAPL) using treasury yields and historical log-returns.
* Quantitative comparison of the PINN-generated surface against a cross-sectional snapshot of AAPL call options.
* Draft a 2–3 page final report in LaTeX, adhering to the course's structural requirements and citing foundational literature.

### Update ###

* We have completed the implementation of a Parametric Black-Scholes PINN using float64 double-precision on the NVIDIA RTX 6000. The high-performance hardware allowed us to maintain numerical stability while calculating the second-order derivative (Gamma) in the PDE residual.
* We have also completed the data acquisition phase using the yfinance API. We successfully extracted 77 distinct data points for AAPL call options across three different time horizons.
* We have also completed the calibration of the model to \sigma = 0.14 and r = 0.04, resulting in a high degree of alignment between the PINN price surface and market mid-prices.
* We have also completed a partial draft of the final paper, including the mathematical methodology, hardware setup, and a discussion of the results.
* We are stuck on minor LaTeX bibliography issues where references were not appearing correctly; however, we have resolved this by ensuring all entries are properly cited within the text.

### Requests ###
I would like to request a brief meeting (5 to 10 minutes) before the final submission deadline on May 1st. I just want to ensure the paper structure aligns with the conference-style requirements and that the "Social Implications" section properly identifies the key stakeholder tensions.

------

## Post-Week 4 Planning

### Goals ###

* Our last Lab meeting is April 29th
* Experimental work should ideally complete a week ahead of time so you can focus on writing
* Paper due on last day of classes (May 1st)
* Work on Final presentation, to be delivered during final exam slot
