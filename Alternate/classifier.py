from lexical import *

def classifier(url):
    # Lexical feature array
    raw_features=[                          # Feature Number
        url_length(url),                    # F01
        special_chars(url),                 # F02
        ratio_special_chars(url),           # F03
        token_count(url),                   # F04
        Presence_of_IP(url),                # F05
        suspicious_word_count(url),         # F06
        subdomain_length(url),              # F07
        domain_token_count(url),            # F08
        query_variables_count(url),         # F09
        max_length_variable(url),           # F10
	max_length_variable_value(url),     # F32
        countdelim(url),                    # F11
        countSubDomain(url),                # F12
        alphabet_count(url),                # F13
        digit_count(url),                   # F14
        countQueries(url),                  # F15
        countdots(url),                     # F16
        count_at_symbol(url),               # F17
        argument_length(url),               # F18
        isPresentHyphen(url),               # F19
        isPresentAt(url),                   # F20
        countSubDir(url),                   # F21
        scheme_http_or_not(url),            # F22
        path_length(url),                   # F23
        directory_length(url),              # F24
        sub_directory_special_count(url),   # F25
        sub_directory_tokens_count(url),    # F26
        filename_length(url),               # F27
        port_number(url),                   # F28
        blacklisted_word_present(url),      # F29
        longest_token_path(url),            # F30
        hyphens_instead_dots_domain(url),   # F31
    ]
    return raw_features

# Example
# print(classifier("www.google.com"))
