#include "sketch/hll.h"
#include <fstream>
#include <omp.h>
#include <iostream>
#include <seqan/seq_io.h>
#include <sketch/include/metrictime2.hpp>

uint64_t canonical_kmer (uint64_t kmer, uint k = 31)
{
	uint64_t reverse = 0;
	uint64_t b_kmer = kmer;

	kmer = ((kmer >> 2)  & 0x3333333333333333UL) | ((kmer & 0x3333333333333333UL) << 2);
	kmer = ((kmer >> 4)  & 0x0F0F0F0F0F0F0F0FUL) | ((kmer & 0x0F0F0F0F0F0F0F0FUL) << 4);
	kmer = ((kmer >> 8)  & 0x00FF00FF00FF00FFUL) | ((kmer & 0x00FF00FF00FF00FFUL) << 8);
	kmer = ((kmer >> 16) & 0x0000FFFF0000FFFFUL) | ((kmer & 0x0000FFFF0000FFFFUL) << 16);
	kmer = ( kmer >> 32                        ) | ( kmer                         << 32);
	reverse = (((uint64_t)-1) - kmer) >> (8 * sizeof(kmer) - (k << 1));

	return (b_kmer < reverse) ? b_kmer : reverse;
}

double jaccard_index (uint u_card, uint i_card)
{
	if (i_card > u_card) return 0;
	return i_card / double (u_card);
}

double compute_distance (std::shared_ptr<sketch::hll_t> s_1, std::shared_ptr<sketch::hll_t> s_2, size_t c_1, size_t c_2)
{
	double dist = 0;
	size_t j = s_1->jaccard_index (*s_2);
	size_t inter = (c_1 + c_2) - j;
	dist = jaccard_index (j, inter);
	//std::cout<<" dist "<<dist<<std::endl;

	return s_1->jaccard_index (*s_2);
}

void estimate_file (std::shared_ptr<sketch::hll_t> s, std::string filename, int k)
{
	seqan::SeqFileIn seqFileIn;
	if (!open(seqFileIn, filename.c_str ()))
	{
		std::cerr << "ERROR: Could not open the file " << filename << ".\n";
		return;
	}

	seqan::CharString id;
	seqan::IupacString seq;

	while (!atEnd (seqFileIn))
	{
		try {
			seqan::readRecord(id, seq, seqFileIn);
		}
		catch (seqan::ParseError a) {
			break;
		}

		uint64_t kmer = 0;
		size_t bases = 0;
		for (size_t i = 0; i < length(seq); ++i)
		{
			uint8_t two_bit = 0;//(char (seq[i]) >> 1) & 0x03;
			bases++;

			switch (char (seq[i]))
			{
				case 'A': two_bit = 0; break;
				case 'C': two_bit = 1; break;
				case 'G': two_bit = 2; break;
				case 'T': two_bit = 3; break;
				case 'a': two_bit = 0; break;
				case 'c': two_bit = 1; break;
				case 'g': two_bit = 2; break;
				case 't': two_bit = 3; break;
				// Ignore kmer
				default: two_bit = 0; bases = 0; kmer = 0; break;
			}

			kmer = (kmer << 2) | two_bit;
			kmer = kmer & ((1ULL << (k << 1)) - 1);

			if (bases == k)
			{
				//s->add (XXH3_64bits ((const void *) &kmer, sizeof (uint64_t)));
				s->addh(canonical_kmer (kmer));
				bases--;
			}
		}
	}
	close (seqFileIn);
}

//void load_file_list (std::vector<std::string> & files, std::string & list_file, std::string path = "./")

void load_file_list (std::vector<std::string> & files, std::string & list_file, std::string path = "")
{
	std::string line;

	if (list_file.empty ())
	{
		std::cerr << "No input file provided\n";
		exit (-1);
	}

	std::ifstream file (list_file);

	if (!file.is_open ())
	{
		std::cerr << "No valid input file provided\n";
		exit (-1);
	}

	while (getline (file, line)) files.push_back (path + line);
	file.close();
}

struct data_s {
	std::shared_ptr<sketch::hll_t> sketch;
	std::string filename;
	double card;
};

int main(int argc, char *argv[])
{
	std::vector<std::string> files;

	std::string list_file = "";
	uint threads = 8;
	const uint k = 31;
	const uint sketch_bits = 14;
	float threshold = 0.9;
	bool build_sketches = false;
	std::string filename_matrix;
	std::ofstream o_matrix;

	char c;

	while ((c = getopt(argc, argv, "l:t:h:o:b")) != -1)
	{
		switch (c) {
			case 'l':
				list_file = std::string (optarg);
			break;
			case 't':
				threads = std::stoi (optarg);
			break;
			case 'h':
				threshold = std::stof (optarg);
			break;
			case 'o':
				filename_matrix = optarg;
			break;
			case 'b':
				build_sketches = true;
			break;
			default:
			break;
		}
	}

	TIMERSTART(total)

	// Inicializar variables:
	omp_set_num_threads (threads);
	load_file_list (files, list_file);

	if (!filename_matrix.empty()) o_matrix.open(filename_matrix, std::ios::out);
	std::ostream & output_matrix = (filename_matrix.empty()) ? std::cout : o_matrix;
	std::string out[files.size ()];

	std::vector<std::pair<std::string, double>> card_name (files.size ());
	std::map<std::string, std::shared_ptr<sketch::hll_t>> card_hll;

	for (size_t i_processed = 0; i_processed < files.size (); ++i_processed)
	{
		std::string filename = files.at (i_processed);
		card_hll[filename] = std::make_shared<sketch::hll_t> (sketch_bits);
	}

	TIMERSTART(construccion)
	
	// Cargar sketches desde archivos.
	// card_hll: hll's de tamaño p=14

	// card_name: Pares (nombre, estimado) para card_hll

	if (build_sketches)
	{
		#pragma omp parallel for schedule(dynamic)
		 for (size_t i_processed = 0; i_processed < files.size (); ++i_processed)
		 {
			std::string filename = files.at (i_processed);
			estimate_file (card_hll[filename], filename, k);
			card_hll[filename]->write (filename + ".hll");

			// c: estimación dada por hll de p=14
			auto c = card_hll[filename]->report ();
			// card_name: Pares (nombre, estimado) para card_hll
			card_name.at (i_processed) = std::make_pair (filename, c);
		}
	}
	else
	{
		#pragma omp parallel for schedule(dynamic)
		for (size_t i_processed = 0; i_processed < files.size (); ++i_processed)
		{
			std::string filename = files.at (i_processed);
			card_hll[filename] = std::make_shared<sketch::hll_t>(filename + ".hll");

			auto c = card_hll[filename]->report ();
			card_name.at (i_processed) = std::make_pair (filename, c);
		}
	}

	// Ordenamos los pares de card_name según la estimación (hll's de tamaño p=14).
	std::sort (card_name.begin (), card_name.end (),
	           [](const std::pair<std::string, double> &x,
	              const std::pair<std::string, double> &y)
	{
		return x.second < y.second;
	});


	TIMERSTOP(construccion)

	// Empezamos a realizar las comparaciones
	TIMERSTART(comparaciones)

	#pragma omp parallel for schedule(dynamic)
	for (size_t i_processed = 0; i_processed < card_name.size () - 1; ++i_processed)
	{
		std::string fn1 = card_name[i_processed].first;
		size_t e1 = card_name[i_processed].second;
		std::string out_str;

		size_t k = i_processed + 1;
		for (; k < card_name.size (); ++k)
		{
			size_t e2 = card_name[k].second;
			if (e2 == 0) continue;

			double fraction = (double)e1 / e2;
			if (fraction < threshold) break;

			std::string fn2 = card_name[k].first;
			double jacc14 = card_hll[fn1]->jaccard_index (*card_hll[fn2]);
			if(jacc14 >= threshold){
				out_str += fn1 + " " + fn2 + " " + std::to_string(jacc14) + "\n";
			}
		}
		out[i_processed] = out_str;
	}

	TIMERSTOP(comparaciones);

	for (size_t i_processed = 0; i_processed < card_name.size (); ++i_processed)
	{
		output_matrix << out[i_processed];
	}

	TIMERSTOP(total)

	return 0;
}

