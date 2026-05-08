import pickle
import plotly.graph_objects as go
import sys
from pathlib import Path


def main():
	if len(sys.argv) < 2:
		raise ValueError('Usage: python tools/plot_3D.py <iteration>')

	iteration = sys.argv[1]
	pickle_path = Path('results') / '1st_order_3D' / '0' / 'images' / f'primitive_0_iter_{iteration}.pickle'

	with pickle_path.open('rb') as file_obj:
		plot_data = pickle.load(file_obj)

	fig = go.Figure(data=plot_data['3D_plot'])
	fig.show()


if __name__ == '__main__':
	main()
