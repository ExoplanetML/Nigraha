import os, sys

if __name__ == '__main__':
        in_path = sys.argv[1]
        out_file = sys.argv[2]
        with open(out_file, 'w') as ofp:
                print(f'tic_id,Sector,Filename', file=ofp)
                files = os.listdir(in_path)
                for f in files:
                        parts = f.split('-')
                        if len(parts) < 3:
                                continue
                        sector, lc = int(parts[1][1:]), int(parts[2])
                        print(f'{lc},{sector},{f}', file=ofp)
