#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <map>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlib/optimization.h>
#include <errno.h>
#include <omp.h>

#define MAXLEN 14
#define TIME_MEASURE milliseconds
#define MAX_ITERATIONS 100
#define NUM_THREADS 4

typedef dlib::matrix<double,0,1> column_vector;

struct dataset
{
	int index;
	std::vector<std::string> sentence ;
	std::string prev_val;
	std::string prev_prev_val;
};

std::vector <dataset> global_d;
std::vector <std::string> global_tags;

std::vector <std::string> global_all_y;

int no_of_calls_cost;
int no_of_calls_gradient;
long int tot_time_cost;
long int tot_time_gradient;
double *global_param;

int (**global_fn_ptr)(int,int);
int global_dim=MAXLEN;

double global_reg=0;

int global_index; 
int global_num_examples;

std::vector< std::vector<double> > global_dataset;


std::set<std::string> prepositions {"about","above","across","after",\
	"against","along","among","around","at","before","behind","below",\
	"beneath","beside","between","by","down","during","except","for","from",\
	"in","inside","instead of","into","like","near","of","off","on","onto",\
	"out of","outside","over","past","since","through","to","toward","under",\
	"underneath","until","up","upon","with","within","without"};

std::set<std::string> determiners {"a","an","the","this","that","these","those","my","your",\
										"her","his","its","our","their"};
std::set<std::string> existance {"is","was","were","has"};
std::set<std::string> personal_pronns {"i", "you","he", "she", "it", "they", "we"};
std::set<std::string> modal_verb {"can", "could", "may", "might", "must", "ought",\
						 "shall", "should", "will", "would"};

std::set<std::string> conjunctions {"and","or","but","nor","so","for", "yet"};


#if 1
std::map <std::string,int> global_tag_index;
std::vector <std::string> global_words;
std::set <std::string> global_all_words;
std::map <std::string,int> global_word_index;
#endif


void split_string(std::string s, std::string* s3 , std::string* s4)
{
	std::string s1,s2;
	int delim=s.find("/");
	int len = s.size();
	s1=s.substr(0,delim);
	s2=s.substr(delim+1,len-1);
	*s3=s1;
	*s4=s2;
	
}

#if 1
void split_string_space(std::string s, std::vector<std::string>& x)
{
	std::string s1,s2;
	int delim,len;
	
	while((delim=s.find(" ")) >= 0)
	{
	len = s.size(); 
	s1=s.substr(0,delim);
	s=s.substr(delim+1,len-1);
	x.push_back(s1);
	
	}
	x.push_back(s);
	
}
#endif

void create_dataset(char *s,char *s1)
{	
	std::vector<std::string> untagged ; 
	std::vector<std::string> tags;
	std::vector<dataset> d;
	std::string line,prev,prev_prev ;
	std::string word1,word2;
	int count=0 , word_index =0 , half_index =0,count1=-1;
	std::ifstream myfile,myfile1;
	#if 1
	int len_sentence ;
	#endif

	myfile.open(s);
	if (myfile.is_open())
  	{
    		while ( getline (myfile,line) )
    		{   	
			++count;
			untagged.push_back(line);		
			//std::cout << line << '\n';
    		}
    		myfile.close();
  	}

  	else std::cout << "Unable to open file";


	myfile1.open(s1);
	prev="";
	prev_prev="";
	
	if (myfile1.is_open())
  	{
		
    		while ( getline (myfile1,line) )
    		{		
			
			if(half_index==2)
			{	word_index =0;
				half_index=0;
				prev="";
				prev_prev="";
				++count1;
			}
			if(line=="")
				++half_index;
			
			else
			{
				
				#if 1
				std::vector<std::string> word_sentence ;
				split_string_space(untagged[count1],word_sentence);
				len_sentence=word_sentence.size();
				#endif
				
				d.push_back(dataset());
				
				#if 1
				if(word_index>0)
					d[d.size()-1].sentence.push_back(word_sentence[word_index-1]);
				else
					d[d.size()-1].sentence.push_back("*");

				if(word_index>1)
					d[d.size()-1].sentence.push_back(word_sentence[word_index-2]);
				else
					d[d.size()-1].sentence.push_back("*");

				d[d.size()-1].sentence.push_back(word_sentence[word_index]);

				if(word_index<len_sentence-1)
					d[d.size()-1].sentence.push_back(word_sentence[word_index+1]);
				else
					d[d.size()-1].sentence.push_back("*");

				if(word_index<len_sentence-2)
					d[d.size()-1].sentence.push_back(word_sentence[word_index+2]);
				else
					d[d.size()-1].sentence.push_back("*");

				
//				d[d.size()-1].sentence=untagged[count1];
				#endif
               			d[d.size()-1].index=word_index;
				
				if(prev=="")
					d[d.size()-1].prev_val="*";
				else
				{
					split_string(prev,&word1,&word2);
					d[d.size()-1].prev_val=word2;
					
			 	}
			     	if(prev_prev=="")
					d[d.size()-1].prev_prev_val="*";
				else
				{
					split_string(prev_prev,&word1,&word2);
					d[d.size()-1].prev_prev_val=word2;
					
			 	}
				
			     	++word_index;
				split_string(line,&word1,&word2);
				#if 1
				global_words.push_back(word1);
				#endif
				tags.push_back(word2);
				prev_prev=prev;
				prev=line;
					
				
			}
    		}
    		myfile1.close();
	
	global_d=d;
	global_tags=tags;
	
  	}
  	else std::cout << "Unable to open file";

}

#if 1
void preprocess()
{
	std::map <std::string , int> word_count ;	
	int i,index=0 ;
	std::map<std::string, int>::iterator iter;

	for(i=0;i<global_all_y.size();++i)
	{
		global_tag_index[global_all_y[i]]=index++;
	}

	index=0;

	for(i=0;i<global_words.size();++i)
	{
		word_count[global_words[i]]++;
	}
	
	for(iter=word_count.begin();iter!=word_count.end();++iter)
	{
		if(iter->second)
		{
			global_all_words.insert(iter->first);
			global_word_index[iter->first]=index++;
		}
	}
	//global_dim=word_count.size() * global_all_y.size() ;

}
#endif

int myfunc(int i,int j)
{
	#if 1
	srand(time(NULL));
	double z= ( (double)rand() / RAND_MAX) +0.5;
	return (int)z ;
	#endif
}

/*
*/

double vector_mult(double* a,double* b,int n=MAXLEN)
{
	int i ;
	double sum=0;
	
	for( i=0 ;i<n;++i)
	{
		sum+=a[i]*b[i];

		
	}
	
	return sum;

}
#if 1
void get_features(struct dataset x,std::string y,double* a)
{
	std::string current_word = x.sentence[2];
	for (int i=0;i<MAXLEN;++i)
	{
		a[i] = 0;
	}
	int first_char_ascii = (int)current_word[0];
	// if First char is capitalized and if tag is NNP | NNPS
	if ( (first_char_ascii >=65 && first_char_ascii <=90) && (y.compare(0,3,"NNP") == 0) ) a[0] = 1;

	// if First char is capitalized and if tag is previous tag is NNP | NNPS
	if ( (first_char_ascii >=65 && first_char_ascii <=90) && (x.prev_val == "NNP") ) a[1] = 1;

	// if previous tag is adjective {JJ,JJR,JJS} and correct tag is noun {NN,NNP,NNS,NNPS}
	if ( (x.prev_val.compare(0,2,"JJ")==0) && (y.compare(0,2,"NN") == 0)  ) a[2] = 1;
	
	

	// if previous tag is adverb {RB,RBR,RBS} and current tag is verb {VBD,VBG,VBN,VBP,VBZ}
	if ( (x.prev_val.compare(0,2,"RB")==0) && (y.compare(0,2,"VB") == 0 && y.length() > 2)  ) a[3] = 1;

	
	// Fucking converting to lowercase is 0(n)
	std::string cur_sub ;
	std::string cur_sub1;
	cur_sub.resize(current_word.size());
	std::transform(current_word.begin(),current_word.end(),cur_sub.begin(),tolower);
	cur_sub1 = cur_sub.substr(0,2);

	// his f5 function will make an O(n) impact here. 
	// all words in his  list except underneath will go exhibit same behaviour here
	
	if ( (prepositions.find(cur_sub) != prepositions.end()) && y == "IN" ) a[4] = 1; 
	

	// if current word starts with WH and is tagged as {WDT,WP,WRB}
	if ( cur_sub1 == "wh" && (y.compare(0,1,"W") == 0) ) a[5] = 1;

	// All conjunctions. once again trying to avoid another O(n) search here.
	//if (current_word.size() <=3 && y == "CC") a[6] = 1;
	if( conjunctions.find(cur_sub) != conjunctions.end()  && y == "CC" ) a[6]=1;

	// this is probably for digits . his f8 had float(s) with s not defined in local scope
	if ( std::all_of(current_word.begin(),current_word.end(), ::isdigit) && y == "CD" ) a[7] = 1;

	// For determiners.
	
	if ((determiners.find(cur_sub) != determiners.end() ) && y == "DT") a[8] = 1;

	// Existential feature: if current word is there and next word is {is|was..}
	std::string next_word (x.sentence[3]);
	next_word.resize(next_word.size());
	std::transform(x.sentence[3].begin(),x.sentence[3].end(),next_word.begin(),tolower);
	//std::cout<<cur_sub<<" FUCK "<<x.sentence[3]<<" LL:"<<next_word<<"\n";
	if ( (cur_sub == "there" && existance.find(next_word) != existance.end()) && y == "EX" )
		 a[9] =1;

    // Personal pronoun: i,we,she etc
	
	if ((personal_pronns.find(cur_sub) != personal_pronns.end() ) && y == "PRP") a[10] = 1;

	// A feature for TO 
	// note that current_word has been transformed to lowercase a lil while before
	if (cur_sub == "to" && y == "TO" ) a[11] = 1;

	// The next feature is for foreign word. 
	//pretty sure C++ wouldve crashed if there was a foreign word
	// Python is unicode c++ is all ASCII
	if (y == "FW") a[12] = 1; // This can literally never happen

	// modal verb

	if ( (modal_verb.find(cur_sub) !=  modal_verb.end())  && y == "MD") a[13] = 1;

}
#endif

double cost(const column_vector& matrix_vector)
{
	++no_of_calls_cost;
	if (PRINT_DEBUG_LEVEL == 2) std::cout<<"Cost Call no: "<<no_of_calls_cost<<"\n";	
	auto t1 = std::chrono::high_resolution_clock::now();
	double sum=0,cost,emperical=0,expected =0,dp,temp;
	int i,j;
	double params[MAXLEN];
	for (i=0;i<MAXLEN;i++)
	{
		params[i] = matrix_vector(0,i);
		if (params[i] != params[i]){
			params[i] = 0.0;
			
		}

	}
	global_param=params;
	for (i=0;i<MAXLEN;++i)
	{
		//params[i]=params[i]*params[i];
		sum+=params[i]*params[i];
	}
	double reg_term= 0.5*global_reg*sum;

	#pragma omp parallel for private(i,j,dp,temp) reduction(+:emperical) reduction(+:expected) 
	for(i=0;i<global_index;++i)
	{
		double get_feat[MAXLEN];
		#if 1
		get_features( global_d[i] , global_tags[i] , get_feat ) ;
		#endif
		dp=vector_mult( get_feat , params , MAXLEN);
		emperical+=dp;
		temp=0;
		for(j=0;j<global_all_y.size();++j)
		{
			#if 1
			get_features( global_d[i] , global_all_y[j] , get_feat ) ;
			#endif
			dp=vector_mult(get_feat , params , MAXLEN);
			temp+=exp(dp);
			if (errno == ERANGE){

				std::cout<<strerror(errno)<<"\n";
				std::cout<<"Temp "<<temp<<" dp "<<dp<<"\n";
				for (i =0;i<MAXLEN;++i)
				{
					std::cout<<get_feat[i]<<" - "<<params[i]<<"\n";
				}
				exit(0);
			}
		}
		expected += log(temp);
		if (std::isnan(expected))
		{
			std::cout<<strerror(errno)<<"\n";
			exit(0);
		}

	}
	cost = (expected - emperical) + reg_term ;
	std::cout<<  "emperical " << emperical<< " reg " << global_reg << " expected "<< expected << std::endl;	

	if (std::isnan(cost))
	{
		cost=0.0;	
	}  // re-normalising false cost computation due to input (or sometimes) dlib::error
	auto t2 = std::chrono::high_resolution_clock::now();
	tot_time_cost += std::chrono::duration_cast<std::chrono::TIME_MEASURE> (t2 - t1).count();
	return cost;
}

column_vector gradient(const column_vector& matrix)
{
	++no_of_calls_gradient;	
	if (PRINT_DEBUG_LEVEL == 2) std::cout<<"Gradient Call no: "<<no_of_calls_gradient<<"\n";
	auto t1 = std::chrono::high_resolution_clock::now();
	double param[MAXLEN];
	std::vector <double> gradient ;
	for (int i=0;i<MAXLEN;++i)
	{
		param[i] = matrix(0,i);
		if (std::isnan(param[i])) param[i] = 0.0;

	}
	if (GRADIENT_VERSION ==3){
	global_param=param;
	
	int i,k,j,l;
	double emperical,expected,mysum ,denominator,numerator,div;
	double fx_yprime[MAXLEN],feats[MAXLEN];
	double normalizer=0.0;
	
	#if 2
	std::vector <double> all_sum;
	int abc;
	#endif

	std::cout << "dim" << global_dim << "num eg" << global_num_examples << "all_y" << global_all_y.size() << std::endl ;
	for(abc=0;abc<global_dim;++abc)
		all_sum.push_back(0);

	double m0=0,m1=0,m2=0,m3=0,m4=0,m5=0,m6=0,m7=0,m8=0,m9=0,m10=0,m11=0,m12=0,m13=0;


	#pragma omp parallel for private(j,i,l,denominator,numerator,div,fx_yprime,feats) firstprivate( global_param) 
	for(j=0;j<global_num_examples;++j)
	{
		denominator=0.0;
		div = 0;		
		for(l=0;l<global_all_y.size();++l)
		{
		#if 1
		get_features(global_d[j] , global_all_y[l] , feats);	
		#endif					
		denominator+=exp(vector_mult(global_param,feats));
		if (std::isnan(denominator)) denominator = 0.0; // re-normalising error computation based on Ratnaparki-1997 	
		}	
	
		for(i=0;i<global_all_y.size();++i)
		{
		#if 1
		get_features(global_d[j], global_all_y[i] , fx_yprime);
		#endif
		numerator=0.0;
		numerator=exp(vector_mult(global_param,fx_yprime));
		if (std::isnan(numerator)) numerator=0.0; // re-normalising error computation based on Ratnaparki-1997 
		div=numerator/denominator;
		
		#pragma omp critical
		for(int abc=0;abc<global_dim;++abc)
			all_sum[abc] += div * fx_yprime[abc] ;
	
		}
	
	}			
	
	
	for(k=0;k<global_dim;++k)
	{
		emperical = 0.0;
		expected = 0.0;
		for(j=0;j<global_dataset.size();j++)
		{
			emperical += global_dataset[j][k];
		}
		std::cout<<  "emperical " << emperical<< " reg " << global_reg << " expected "<< all_sum[k] << std::endl;
			gradient.push_back(all_sum[k] - emperical + global_reg);
	}
	}
	if (GRADIENT_VERSION == 2){	
	global_param=param;
	int i,k,j,l;
	double emperical,expected,mysum ,denominator,numerator,div;
	double fx_yprime[MAXLEN],feats[MAXLEN];
	double normalizer=0.0;
	for(k=0;k<global_dim;++k)
	{
		emperical = 0.0;
		expected = 0.0;
	
		for(j=0;j<global_dataset.size();j++)
		{
			emperical += global_dataset[j][k];
		}
		
		for(j=0;j<global_num_examples;++j)
		{
			denominator=1.0;
			for(l=0;l<global_all_y.size();++l)
			{
					#if 1
					get_features(global_d[j] , global_all_y[l] , feats);	
					#endif					
					denominator+=exp(vector_mult(global_param,feats));
					if (std::isnan(denominator)) denominator =0.45; // re-normalising based onRatnaparki-1997 

			}

			mysum=0.0;
			for(i=0;i<global_all_y.size();++i)
			{
				#if 1
				get_features(global_d[j], global_all_y[i] , fx_yprime);
				#endif
				
				numerator=0.0;
				numerator=exp(vector_mult(global_param,fx_yprime));
				if (std::isnan(numerator)) numerator = 0.54;// re-normalising error computation based on Ratnaparki-1997 

				div=numerator/denominator;
				
				mysum += div * fx_yprime[k] ;
			}
			
			expected += mysum ;
		}
			gradient.push_back(expected - emperical + global_reg);
	}
	}
	column_vector ret_param(MAXLEN);
	std::cout<<"[\t";
	for (int i=0;i<MAXLEN;++i)
	{
		ret_param(0,i) = gradient[i];
		std::cout<<gradient[i]<<"\t";

	}
	
	auto t2 = std::chrono::high_resolution_clock::now();
	tot_time_gradient += std::chrono::duration_cast<std::chrono::TIME_MEASURE> (t2 - t1).count();
	std::cout<<"\t]\n";
	return ret_param;
}


column_vector train()
{
	preprocess();
	column_vector initial_guess(MAXLEN);
	initial_guess = {0.0};	
	#if 1
	dlib::find_min(dlib::lbfgs_search_strategy(MAXLEN),
			dlib::objective_delta_stop_strategy(1e-07,MAX_ITERATIONS).be_verbose(),
			cost,gradient,initial_guess,-1);
	#endif
	
	//cost(initial_guess);
	//gradient(initial_guess);
	return initial_guess;
}

void memm()
{

	
	
	#if 1
	int i,j,k ;
	for(i=0;i<global_d.size();++i)
	{
		for (j=0;j<global_all_y.size();++j)
		{
			if(global_tags[i]==global_all_y[j])
			{
				double val[MAXLEN];
				std::vector<double> vals;
				#if 1
				get_features(global_d[i], global_all_y[j] ,val);
				#endif
				for(k=0;k<global_dim;++k)
					vals.push_back(val[k]);
		
				global_dataset.push_back(vals);
			}
		}
	}	


	#endif

	
}



int main()
{

	char filename1[25]="tagged_sent_sample";
	char filename2[25]="untagged_sent_sample";
	create_dataset(filename2,filename1);
	

	int i;

	int (*fn_ptr[MAXLEN])(int,int);
	
	for(i=0;i<MAXLEN;++i)
	{
		fn_ptr[i]=&myfunc;
	}
	
	global_fn_ptr=fn_ptr;
	
	
	global_num_examples=global_d.size();
	std::set <std::string> all_y;
	global_index=(global_d.size()<global_tags.size())?global_d.size():global_tags.size();
	for(i=0;i<global_tags.size();++i)
	{
		all_y.insert(global_tags[i]);

	}
	global_all_y.assign(all_y.begin(), all_y.end());

	column_vector trained_model(MAXLEN);
	trained_model = {0.0};
	for (int i=0;i<=trained_model.nc()+1;i++) std::cout<<trained_model(0,i)<<"\t";
	memm();
	trained_model = train();
	std::string time_unit = " milliseconds\n";
	if (PRINT_DEBUG_LEVEL >= 1){
	std::cout<<"For "<<no_of_calls_cost<<" calls to cost, total time taken is "<<tot_time_cost<<time_unit;
	std::cout<<"Per call, avg time is "<<(tot_time_cost/no_of_calls_cost)<<time_unit;
	std::cout<<"For "<<no_of_calls_gradient<<" calls to gradient, total time taken is "<<tot_time_gradient<<time_unit;
	std::cout<<"Per call, avg time is "<<(tot_time_gradient/no_of_calls_gradient)<<time_unit;
	}
	std::cout<<"Trained model\n";
	for (int i=0;i<trained_model.nr();i++) std::cout<<trained_model(0,i)<<"\t";
	std::cout<<trained_model.nr()<<" : "<<trained_model.nc()<<"\n";
	std::cout<<strerror(errno)<<"\n";
	
	return 0;

}


