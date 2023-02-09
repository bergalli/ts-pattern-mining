import glob
import json
import os
import re
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Tuple
from zipfile import ZipFile
from pathlib import Path
import pandas as pd
import tqdm


def main(downloaded_folder: str, convert_to_parquet: bool):
    postproc_folder = os.path.join(os.path.split(downloaded_folder)[0],
                                   os.path.split(downloaded_folder)[1] + '_postproc')

    unzip_files(source_root_folder=downloaded_folder, dest_root_folder=postproc_folder)

    symbols_spec = json.load(open(os.path.join('external', 'binance-public-data', 'data', 'symbols.json'), 'r'))
    symbols_spec = symbols_spec['symbols']
    assert len(symbols_spec) == len(set(map(lambda x: x['symbol'], symbols_spec))), 'found duplicated symbols'

    process_files(postproc_folder, symbols_spec, convert_to_parquet)

    print(f'Data processed and saved in {postproc_folder}')


def process_files(postproc_folder, symbols_spec, convert_to_parquet):
    csv_files = glob.glob(os.path.join(postproc_folder, '**/*.csv'), recursive=True)
    for csv_f in tqdm.tqdm(csv_files, desc='Processing files'):
        columns, symbol = get_columns_and_symbol(csv_f, postproc_folder)

        df = pd.read_csv(csv_f, names=columns)

        # convert timestamp from milliseconds to seconds
        df['timestamp'] = df['timestamp'] / 1000

        df['symbol'] = symbol
        for spec in symbols_spec:
            if spec['symbol'] == symbol:
                df['base_asset'] = spec['baseAsset']
                df['quote_asset'] = spec['quoteAsset']
                break

        if convert_to_parquet:
            df.to_parquet(os.path.splitext(csv_f)[0] + '.parquet')
            os.remove(csv_f)
        else:
            df.to_csv(csv_f)


def get_columns_and_symbol(csv_f: str, postproc_folder: str) -> Tuple[pd.Index, str]:
    def get_folder_pattern(dkind):
        return os.path.join(postproc_folder, 'spot', '.*', dkind, '')

    def get_file_symbol(dkind, filepath):
        return Path(re.split(get_folder_pattern(dkind), filepath)[-1]).parts[0]

    def rename_if_key(val, rename_mapper):
        if val in rename_mapper:
            return rename_mapper[val]
        else:
            return val

    if re.match(get_folder_pattern('klines'), csv_f):
        columns = pd.Index(
            ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
             'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        )
        columns = columns.map(lambda x: rename_if_key(x,
                                                      {'Open time': 'timestamp',
                                                       'Open': 'open',
                                                       'High': 'high',
                                                       'Low': 'low',
                                                       'Close': 'close',
                                                       'Volume': 'volume',
                                                       'Close time': 'timestamp_close'}))
        symbol = get_file_symbol('klines', csv_f)

    elif re.match(get_folder_pattern('trades'), csv_f):
        columns = pd.Index(
            ['trade Id', ' price', ' qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
        )
        symbol = get_file_symbol('trades', csv_f)

    elif re.match(get_folder_pattern('aggTrades'), csv_f):
        columns = pd.Index(
            ['Aggregate tradeId', 'Price', 'Quantity', 'First tradeId', 'Last tradeId', 'Timestamp',
             'Was the buyer the maker', 'Was the trade the best price match'])
        symbol = get_file_symbol('aggTrades', csv_f)

    else:
        raise

    return columns, symbol


def unzip_files(source_root_folder: str, dest_root_folder: str):
    zipped_files = glob.glob(os.path.join(source_root_folder, '**/*.zip'), recursive=True)
    # Create a ZipFile Object and load sample.zip in it
    for zip_f in tqdm.tqdm(zipped_files, desc='Unzipping files'):
        with ZipFile(zip_f, 'r') as zipObj:
            save_folder = os.path.split(zip_f.replace(source_root_folder, dest_root_folder))[0]
            # Extract all the contents of zip file in different directory
            zipObj.extractall(save_folder)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--downloaded-folder', dest='source_root_folder', type=str,
                        help='Root folder containing the data')
    parser.add_argument('--to-parquet', dest='convert_to_parquet', action='store_true')

    args = parser.parse_args(sys.argv[1:])
    main(args.source_root_folder, args.convert_to_parquet)
