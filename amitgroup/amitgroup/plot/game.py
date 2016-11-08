from __future__ import division, print_function, absolute_import
import numpy as np

class PlottingWindow(object):
    """
    A simple `pygame <http://www.pygame.org/>`__ wrapper that allows for real-time plotting with minimal code injection.

    Pygame prefers to be run in the main thread, so you are in charge of maintaing a main loop that calls :func:`tick`. 
    However, for the development of iterative algorithms, this is appropriately the same as the main algorithm loop.

    Parameters
    ----------
    figsize : tuple
        Figure size, with a similar scale as `matplotlib <http://matplotlib.sourceforge.net>`__ uses. Tuple of size 2.
    subplots : tuple
        Number of subplots, with rows specified first and then columns. Tuple of size 2.
    caption : str
        Set the name of the plotting window.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import numpy as np
    
    Plot an evolving 2D function.

    >>> plw = ag.plot.PlottingWindow() # doctest: +SKIP
    >>> x, y = np.mgrid[0:1:100j, 0:1:100j]
    >>> t = 0.0
    >>> while plw.tick():
    ...     im = np.sin(x + t) * np.cos(10*y + 10*t)
    ...     plw.imshow(im, limits=(-1, 1))
    ...     t += 1/60. # doctest: +SKIP
    """
    def __init__(self, figsize=(8, 6), subplots=(1, 1), caption="Plotting Window"):
        import pygame

        pygame.init() 
        #if not pygame.font: print('Warning, fonts disabled')
        self._window_size = tuple(map(lambda x: x*80, figsize))
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption(caption)
        pygame.mouse.set_visible(1)

        self._clock = pygame.time.Clock()

        self._background = pygame.Surface(self._screen.get_size())
        self._background = self._background.convert()
        self._background.fill((130, 130, 130))

        i = 0
        self._quit = False
        self.subplots(subplots)

    def subplots(self, shape):
        """
        Change the number of subplots. It is normally more convenient to set this through an argument in the constructor.
        """
        assert len(shape) == 2, "Subplots shape must have two elemements"
        self._subplots = shape

    def clear(self):
        """
        Manually clear the screen. If :func:`tick` is called with its default settings, a separate call to this function is not needed.
        """
        self._screen.blit(self._background, (0, 0))

    def tick(self, clear=True, fps=0):
        """
        Handle input and clear the screen. 

        If this function returns `False`, you should terminate your main loop. 
        
        Parameters
        ----------
        fps : int
            If nonzero, then it will enforce the given frame rate. For instance, set to 60, if you want 60 a maximum of 60 frames per seconds to be displayed.
        """
        import pygame
        import pygame.locals as pylocals
        pygame.display.flip()
        if fps > 0:
            self._clock.tick(fps)

        if self._quit:
            return False
        #clock.time(60)  
        for event in pygame.event.get():
            if event.type == pylocals.QUIT or \
               event.type == pylocals.KEYDOWN and event.key == pylocals.K_ESCAPE:
                self._quit = True
                pygame.display.quit()
                return False 


        if clear:
            self.clear()

        return True 

    def _anchor_and_size(self, subplot):
        pad = 10 
        p = (subplot%self._subplots[1], subplot//self._subplots[1])
        size = (self._window_size[0]//self._subplots[1], self._window_size[1]//self._subplots[0])
        anchor = (p[0] * size[0], p[1] * size[1])

        return (anchor[0]+pad, anchor[1]+pad), (size[0]-2*pad, size[1]-2*pad) 

    def imshow(self, im, limits=(0, 1), subplot=0, caption=None):
        """
        Display an image.
    
        Parameters
        ----------
        im : ndarray
            A 2D array with the image data.
        limits : tuple
            A tuple of size two, that specifies the values of black and white.
        subplot : int
            Zero-based index of subplot.
        caption : str
            Add a textual description of the image.
        """
        import pygame
        assert isinstance(im, np.ndarray) and len(im.shape) == 2, "Image must be a 2D ndarray"
        anchor, size = self._anchor_and_size(subplot)

        # Normalize
        if limits != (0, 1):
            span = (limits[1]-limits[0])
            if span == 0.0:
                return
            im2 = (im-limits[0])/span
        else:
            im2 = im

        im3 = (np.clip(im2.T, 0, 1)*255).astype(np.uint32)
        scr = pygame.Surface(im3.shape)
        pygame.surfarray.blit_array(scr, (im3<<24) + (im3<<16) + (im3<<8) + 0xFF)
        scale = min((size[0])//im.shape[0], (size[1])//im.shape[1])
        scr2 = pygame.transform.scale(scr, (im.shape[0]*scale, im.shape[1]*scale))
        #_screen.blit(scr2, (640//2 - im.shape[0]*scale//2, 0))
        self._screen.blit(scr2, anchor)

        if caption and pygame.font:
            font = pygame.font.Font(None, 16)
            text = font.render(caption, 1, (10, 10, 10))
            self._screen.blit(text, (anchor[0], anchor[1]-10))

    def plot(self, x, y=None, limits='auto', subplot=0):
        """
        Plot some data using a regular solid-line plot.

        Parameters
        ---------- 
        x : ndarray
            The `time` axis. If `y` is None, then this is used as data values instead.
        y : ndarray
            The `data` axis.
        limits : tuple or str
            A tuple of size two, that specifies the range of the `y` values that should be plotted. If set to a string saying ``"auto"``, then it will be automatically determined.
        subplot : int
            Zero-based index of subplot.
        """
        import pygame
        N = len(x) 
        if N < 2:
            return # Just don't draw anything
        anchor, size = self._anchor_and_size(subplot)
        if y is None:
            y = x
            x = np.arange(N)
        x = np.asarray(x)
        y = np.asarray(y)
        if limits == 'auto':
            xlims = x.min(), x.max()
            ylims = y.min(), y.max()
        else:
            xlims = x.min(), x.max()
            ylims = limits
    
        @np.vectorize
        def x2pixel(x0):
            return anchor[0] + size[0]*(x0-xlims[0])/(xlims[1]-xlims[0])
        @np.vectorize
        def y2pixel(y0):
            return anchor[1] + size[1]*(1-((y0-ylims[0])/(ylims[1]-ylims[0])))

        px = x2pixel(x)
        py = y2pixel(y)
        pointlist = zip(px, py)
        pygame.draw.aalines(self._screen, (255, 255, 255), False, pointlist)

        if pygame.font: 
            font = pygame.font.Font(None, 16)
            text = font.render("{0:.1g}/{1:.1g}".format(*ylims), 1, (10, 10, 10))
            self._screen.blit(text, (anchor[0], anchor[1]-10))

    def mainloop(self, fps=60):
        """
        Call this instead or after your own :func:`tick` loop, to keep the window alive. 

        This function will be blocking utnil the user presses Escape or closes the window.
        """
        while self.tick(clear=False, fps=fps):
            pass
