using System;
using HtmlAgilityPack;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using System.Net.Http;
using System.Threading.Tasks;
using System.Net;
using Newtonsoft.Json;

namespace data_miner
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Write("Input number of pages:");
            var noPages = Convert.ToUInt32(Console.ReadLine());

            Console.Write("Input maximun number of answer:");
            var limitAnswer = Convert.ToUInt32(Console.ReadLine());

            Console.Write("Input number of thread:");
            var noThreads = Convert.ToUInt32(Console.ReadLine());

            Console.WriteLine("Loading question");
            List<QA> qas = new List<QA>();
            // links.AddRange(getLinkOnPage(1));
            for (int i = 1; i <= noPages; i++)
            {
                Console.WriteLine($"Loading question on page {i}");
                qas.AddRange(getQuestionOnPage(i));
            }
            Console.WriteLine($"Got {qas.Count} question");

            Console.WriteLine("Loading question's answer");
            LoadAnswer(qas, limitAnswer, noThreads).Wait();
            File.WriteAllText("result.txt", JsonConvert.SerializeObject(qas, Formatting.Indented), Encoding.UTF8);
        }

        static async Task LoadAnswer(List<QA> qas, UInt32 limitAnswer, UInt32 noThreads)
        {
            List<Task> tasks = new List<Task>();
            int i = 0;
            var add = Task.Run(() =>
            {
                while (i < qas.Count)
                {
                    if (tasks.Count < noThreads)
                    {
                        Console.WriteLine($"Loading question's answer {i + 1}/{qas.Count}");
                        lock (tasks)
                        {
                            tasks.Add(qas[i].LoadAnswer(limitAnswer));
                            i++;
                        }
                    }
                }
            });

            var remove = Task.Run(async () =>
            {
                while (i < qas.Count || tasks.Count > 0)
                {
                    if (tasks.Count > 0)
                    {
                        var completed = await Task.WhenAny(tasks);
                        lock (tasks)
                        {
                            tasks.Remove(completed);
                        }
                    }
                }
            });

            await Task.WhenAll(add, remove);

        }

        static IEnumerable<QA> getQuestionOnPage(int page)
        {
            var url = $"https://vnexpress.net/tin-tuc/khoa-hoc/hoi-dap/page/{page}.html";
            var web = new HtmlWeb();
            var doc = web.Load(url);
            var descendants = doc.DocumentNode
            .Descendants("section")
            .Where(section => section.Attributes["class"]?.Value == "sidebar_1")
            .FirstOrDefault()
            .Descendants("article")
            .SelectMany(article => article.Descendants("h3")
                .Select(h3 => h3.Descendants("a").First()));

            return descendants.Select(item => new QA()
            {
                Question = item.InnerText.Trim(),
                Link = item.Attributes["href"].Value
            });
        }

    }

    class QA
    {
        private Regex idRegex = new Regex("((?!-).)+(?=\\.html)");

        [JsonIgnore]
        public string Link { get; set; }

        [JsonIgnore]
        public string Id
        {
            get
            {
                return idRegex.Match(Link).Value;
            }
        }
        public string Question { get; set; }

        public List<string> Answers { get; set; } = new List<string>();

        public async Task LoadAnswer(UInt32 limitAnswer)
        {
            HttpClient client = new HttpClient();
            var response = await client.GetAsync($"https://usi-saas.vnexpress.net/index/get?objectid={Id}&siteid=1000000&categoryid=1003137&sort=like&objecttype=1&limit={limitAnswer}&offset=0");
            var jsonString = await response.Content.ReadAsStringAsync();
            jsonString = WebUtility.HtmlDecode(jsonString);
            dynamic json = Newtonsoft.Json.JsonConvert.DeserializeObject(jsonString);

            if (json.iscomment != 1)
            {
                // no comment no answer
                return;
            }

            Answers.AddRange(((IEnumerable<dynamic>)json.data.items).Select(answer => (string)answer.content));
        }


    }
}
