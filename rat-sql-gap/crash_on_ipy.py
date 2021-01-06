import sys

class ExceptionHook:
    instance = None

    def __call__(self, type, value, tb):
      if self.instance is None:
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
          sys.__excepthook__(type, value, tb)
        else:
          import traceback
          # from IPython.core import ultratb
          # self.instance = ultratb.FormattedTB(mode='Plain',
          #      color_scheme='Linux', call_pdb=1)
          import pudb
          traceback.print_exception(type, value, tb)
          pudb.post_mortem(tb)

sys.excepthook = ExceptionHook()