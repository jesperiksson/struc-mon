# The top level program file
# Imports

# Libraries and modules
import argparse

# Other files
from ProgramRunner import ProgramRunner
from Settings import Settings
import config


def main(): 
    settings = Settings()
    # This parser allows the user to set flags 
    parser = argparse.ArgumentParser(
        description = '',
        epilog = '')
    parser.add_argument(
        '--q', 
        help = 'Dont show progress during training and evaluation',
        action = "store_true",
        #metavar = 'QUIET'
        )
    parser.add_argument(
        '--sensor',
        action = 'store',
        nargs = '*',
        default = [],
        type = str,
        help = 'Which kind of data to use. These are available(keyword-sensor): \nacc - Acceleration[g] \nincl - Inclinations[deg] \nstrain - Strain[mV]'
        )
    parser.add_argument(
        '--preset',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        help = 'Uses a different preset neural net'
        )
    parser.add_argument(
        '--load',
        action='store_true',
        #nargs = 0,
        help = 'If provided the model tries to load an existing neural net'
        )
    parser.add_argument(
        '--name',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        #choices = , TODO: list exixting models
        help='Try to load a model with the current name (given the --load argument is provided)'
        )
    parser.add_argument(
        '--mode',
        action = 'store',
        nargs = 1,
        default = ['standard'],
        type = str,
        help = 'What to perform. Currently implemented:\n train - Allocate all data to training set and train. Load if availabe.\n eval - allocate all data to test and evaluate. Load if available.\n train_and_eval - Split data according to data_split. Train and evaluate. \n plot_normalized - Load data, normalize it and display a violin plot.\n sep_train_eval - Load two separate data sets with different queries. '
        )
    parser.add_argument(
        '--date',
        action = 'store',
        nargs = '+',
        type = str,
        help = 'Dates, from between which to read data. Rquired format is yyyy-mm-dd. \nThe order of them implies \n1: Start date\n2: End/middle date\n 3: End date (in case mode is separate train and eval)'
        )
    args = parser.parse_args()
    ProgramRunner(settings,args)
            

if __name__ == "__main__":
    main()
